for b in tensorflow jax torch; do
  echo "=== $b ==="
  KERAS_BACKEND=$b uv run pytest tests || echo "FAILED: $b"
done