#!/bin/bash
CURRENT_DIR=$(pwd)

echo "Starting Docker container with local files..."

docker run --rm -it \
  -e PYTHONPATH=/app \
  -v "$CURRENT_DIR/execute.py:/app/execute.py" \
  -v "$CURRENT_DIR/stateAvoid:/app/stateAvoid" \
  -v "$CURRENT_DIR/example:/app/example" \
  sa_app /bin/bash -c "python /app/execute.py"