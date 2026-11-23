"""Register both models from the latest compare_qa run and set @champion / @challenger.

    python register_champion.py          # aliases follow the rule's decision (winner tag)
    python register_champion.py --swap   # stretch goal: flip the aliases
"""
import argparse

import mlflow
from mlflow import MlflowClient

import config as C


def latest_summary_run(client):
    exp = client.get_experiment_by_name(C.EXPERIMENT_NAME)
    runs = client.search_runs([exp.experiment_id], filter_string=f"attributes.run_name = 'compare_{C.TASK}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit("No compare run found - run evaluate.py first.")
    return runs[0]


def version_for_run(client, run_id):
    """Return the registry version already created from this run, if any."""
    for mv in client.search_model_versions(f"name='{C.REGISTERED_MODEL_NAME}'"):
        if mv.run_id == run_id:
            return mv
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", action="store_true", help="give @champion to the rule's loser")
    args = ap.parse_args()

    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()
    summary = latest_summary_run(client)
    winner = summary.data.tags["winner"]
    print(f"Summary run {summary.info.run_id}: rule winner = {winner} ({summary.data.tags['winner_hf_model_id']})")

    versions = {}
    for key in ("A", "B"):
        run_id = summary.data.tags[f"{key}_run_id"]
        mv = version_for_run(client, run_id)
        if mv is None:
            model_uri = client.get_run(run_id).data.tags["logged_model_uri"]
            mv = mlflow.register_model(model_uri, C.REGISTERED_MODEL_NAME)
            client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "hf_model_id", C.MODELS[key]["hf_model_id"])
            client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "model_key", key)
            client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "student_usn", C.STUDENT_USN)
        versions[key] = mv.version
        print(f"  {key}: {C.MODELS[key]['hf_model_id']} -> {C.REGISTERED_MODEL_NAME} v{mv.version}")

    champ = winner if not args.swap else ("A" if winner == "B" else "B")
    chall = "A" if champ == "B" else "B"
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", versions[champ])
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", versions[chall])
    client.update_registered_model(C.REGISTERED_MODEL_NAME,
                                   description=f"Extractive QA (SQuAD v2) Champion-Challenger - {C.STUDENT_USN}")
    print(f"@champion   -> v{versions[champ]} ({C.MODELS[champ]['hf_model_id']})")
    print(f"@challenger -> v{versions[chall]} ({C.MODELS[chall]['hf_model_id']})"
          + ("   [SWAPPED for the stretch goal]" if args.swap else ""))


if __name__ == "__main__":
    main()
