# Dedicated Wates plan-generation container.
# Based on the reporting image so it inherits the working generator
# (src/plans/generate_plan.py: Gemini -> SVG -> Aspose -> VSDX), the AlphaTracker
# `api` package, and all deps (aspose-diagram-python, svgwrite, gemini libs).
#
# Build + run on the server (image source-of-truth lives in /opt/Plangenration):
#   docker build -t plangenration:latest /opt/Plangenration
#   docker run -d --name plangenration --restart unless-stopped \
#     --env-file /opt/acorn/.env --network acorn_network \
#     -v /opt/Plangenration/reports:/app/src/output/reports \
#     -v /opt/acorn/config:/app/config:ro \
#     plangenration:latest
FROM acorn_reporting:latest

ARG PLAN_GIT_SHA=unknown
LABEL org.opencontainers.image.source="https://github.com/SuryaKambalaAcorn/AcornPlanGenration"
LABEL org.opencontainers.image.revision="${PLAN_GIT_SHA}"

# Aspose(.NET) finds no compatible ICU in this image -> run .NET in invariant
# globalization mode (no ICU needed). process_plan.py also sets this per-subprocess.
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

# Default production extractor: OpenAI GPT-4o only. The server .env can
# override these, but the image should not silently fall back to Gemini.
ENV OPENAI_VISION_MODEL=gpt-4o
ENV PLAN_LAYOUT_PROVIDERS=openai

# The plans orchestrator (gate -> draw via generate_plan.py -> upload -> status).
COPY process_plan.py /app/src/process_plan.py
COPY generate_plan.py /app/src/plans/generate_plan.py
COPY template.vsdx /app/src/plans/template.vsdx
COPY layout_extractor.py /app/src/utils/layout_extractor.py
COPY vision_client.py /app/src/utils/vision_client.py

# Idle, long-lived container that n8n execs into.
ENTRYPOINT []
CMD ["sleep", "infinity"]
