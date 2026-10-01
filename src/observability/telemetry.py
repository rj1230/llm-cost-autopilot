import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_telemetry(app):
    enabled = os.getenv("AUTOPILOT_OTEL_ENABLED", "false").lower() == "true"
    if not enabled:
        return

    resource = Resource.create({
        "service.name": "llm-cost-autopilot",
        "service.version": os.getenv("AUTOPILOT_VERSION", "1.0.0"),
    })

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter())
    )

    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()