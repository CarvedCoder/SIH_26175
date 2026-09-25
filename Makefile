.PHONY: checkpoint health serve

# Produce the serving checkpoint at outputs/calib_net/gamus_rgb_grad/best.pt
# (no-op if already present; pass FORCE=1 to retrain).
checkpoint:
	scripts/fetch_checkpoint.sh $(if $(FORCE),--force,)

# Start the backend locally and confirm the checkpoint is served
# (curl prints {"status":"ok",...,"model_loaded":true}).
health: checkpoint
	uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 & \
	pid=$$!; sleep 5; \
	curl -fsS http://127.0.0.1:8000/api/v1/health; echo; \
	kill $$pid

serve:
	uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
