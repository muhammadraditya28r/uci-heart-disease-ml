import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


IMAGE_NAME = "heart-disease-api"
CONTAINER_PORT = 8000
HOST_PORT = 8001

HEALTH_URL = f"http://localhost:{HOST_PORT}/health"
PREDICT_URL = f"http://localhost:{HOST_PORT}/predict"

VALID_PAYLOAD = {
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalch": 150,
    "oldpeak": 2.3,
    "sex": "Male",
    "cp": "typical angina",
    "fbs": None,
    "restecg": "lv hypertrophy",
    "exang": 0,
    "slope": "downsloping",
    "ca": 0,
    "thal": "fixed defect",
}

STARTUP_TIMEOUT = 30
POLL_INTERVAL = 0.5


def wait_for_health(container_id: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass

        # Check whether the container has already exited.
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.stdout.strip() == "false":
            logs = subprocess.run(
                ["docker", "logs", container_id],
                capture_output=True,
                text=True,
                check=False,
            )

            raise RuntimeError(
                "Container exited before /health became available.\n"
                f"Container logs:\n{logs.stdout}\n{logs.stderr}"
            )

        time.sleep(POLL_INTERVAL)

    raise RuntimeError(
        f"Application did not become ready within {STARTUP_TIMEOUT} seconds."
    )


def test_prediction() -> None:
    payload = json.dumps(VALID_PAYLOAD).encode("utf-8")

    request = urllib.request.Request(
        PREDICT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        assert response.status == 200

        data = json.loads(response.read().decode("utf-8"))

    assert data["prediction"] in {0, 1}
    assert 0.0 <= data["probability"] <= 1.0


def main() -> int:
    container_id = None

    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "-p",
                f"{HOST_PORT}:{CONTAINER_PORT}",
                IMAGE_NAME,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Docker failed to start container:\n{result.stderr.strip()}"
            )

        container_id = result.stdout.strip()

        wait_for_health(container_id)
        test_prediction()

        print("Smoke test passed.")
        return 0

    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)

        if container_id:
            logs = subprocess.run(
                ["docker", "logs", container_id],
                capture_output=True,
                text=True,
                check=False,
            )

            if logs.stdout or logs.stderr:
                print("\nContainer logs:", file=sys.stderr)
                print(logs.stdout, file=sys.stderr)
                print(logs.stderr, file=sys.stderr)

        return 1

    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
