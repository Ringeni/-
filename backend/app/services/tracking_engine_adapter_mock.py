from datetime import datetime, timezone

from app.schemas.mock import AnalyzeResult, Detection, TrackResult


class TrackingEngineAdapterMock:
    @staticmethod
    def analyze_frames(camera_ids: list[int]) -> AnalyzeResult:
        now = datetime.now(timezone.utc)
        results: list[TrackResult] = []
        for idx, camera_id in enumerate(camera_ids):
            results.append(
                TrackResult(
                    cameraId=camera_id,
                    timestamp=now,
                    detections=[
                        Detection(
                            box=[100 + idx * 15, 120, 240 + idx * 15, 380],
                            className="person",
                            score=0.96,
                            targetId=90021,
                            assocScore=0.98,
                        ),
                        Detection(
                            box=[280 + idx * 10, 160, 400 + idx * 10, 390],
                            className="person",
                            score=0.91,
                            targetId=90022,
                            assocScore=0.95,
                        ),
                    ],
                )
            )

        return AnalyzeResult(results=results, modelTimeMs=45, debugSummary={"mode": "mock"})
