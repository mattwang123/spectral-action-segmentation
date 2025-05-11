import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, List, Optional
import cv2
import os

# Load frames
def load_saved_frames(folder_path: str, resize_dim=(64, 64)) -> List[np.ndarray]:
    """Load PNG frames from a folder, convert to gray, resize & normalise."""
    frame_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".png"))
    frames = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
            frames.append(img.astype(np.float32) / 255.0)

    print(f"Loaded {len(frames)} frames from '{folder_path}'")
    return frames


# ---------------------------------------------------------------------
# Markov‑only predictor
# ---------------------------------------------------------------------
class MarkovStagePredictor:
    """
    Learn a first‑order Markov model P(stage_{t+1}=j | stage_t=i) from a *sequence
    of integer labels*.  Optionally keep one representative picture per stage so
    that predictions can be visualised.
    """
    def __init__(self, n_stages: int, forecast_window: int = 3) -> None:
        self.n_stages = n_stages
        self.forecast_window = forecast_window
        self.transition_matrix: Optional[np.ndarray] = None  # shape (K, K)
        self.stage_examples: List[Optional[np.ndarray]] = [None] * n_stages  # one picture per stage

    # ------------------------
    # fitting / learning
    # ------------------------
    def fit(
        self,
        labels: Sequence[int],
        frames: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        """
        Parameters
        ----------
        labels
            1‑D list/array of stage indices (length = #frames).
        frames
            The original images in the *same order* as `labels`.  If given,
            the first occurrence of each stage is stored for later display.
        """
        labels = np.asarray(labels, dtype=int)
        if frames is not None and len(frames) != len(labels):
            raise ValueError("`frames` and `labels` must be the same length")

        # -- 1. learn transition matrix with Laplace smoothing
        K = self.n_stages
        T = np.full((K, K), 1e-6)  # start with epsilon to avoid zero rows
        for curr, nxt in zip(labels[:-1], labels[1:]):
            T[curr, nxt] += 1
        T /= T.sum(axis=1, keepdims=True)
        self.transition_matrix = T

        # -- 2. store an example frame for each stage (optional)
        if frames is not None:
            for k in range(K):
                idx = np.where(labels == k)[0]
                if idx.size:                          # stage actually appears
                    self.stage_examples[k] = frames[int(idx[0])]

    # ------------------------
    # inference / forecasting
    # ------------------------
    def predict_next_stage(
        self,
        recent_labels: Sequence[int],
        return_frame: bool = False,
    ):
        """
        Predict the most likely next stage from the last `forecast_window`
        observed labels.  Returns (stage_id, confidence [, frame]).
        """
        if self.transition_matrix is None:
            raise RuntimeError("Model not fitted yet")

        window = np.asarray(recent_labels)[-self.forecast_window :]
        vote = np.zeros(self.n_stages)
        for curr in window[:-1]:
            vote += self.transition_matrix[curr]
        pred_stage = int(vote.argmax())
        conf = float(vote[pred_stage] / vote.sum()) if vote.sum() > 0 else 0.0

        if return_frame and self.stage_examples[pred_stage] is not None:
            return pred_stage, conf, self.stage_examples[pred_stage]
        return pred_stage, conf


# ---------------------------------------------------------------------
# quick demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1. load frames (only so we can *see* the prediction – not needed for fitting)
    folder = os.path.join("preprocessed_frames", "shahar_jack")
    frames = load_saved_frames(folder)

    # 2. pretend these are *given* stage labels for every frame
    #    (replace with your own array straight from your clustering step)
    labels = np.loadtxt("shahar_jack_stage_labels.txt", dtype=int)

    # 3. train the Markov model
    predictor = MarkovStagePredictor(n_stages=4, forecast_window=5)
    predictor.fit(labels, frames)  # frames are optional

    print("Learned transition matrix:")
    print(predictor.transition_matrix)

    # 4. forecast
    recent = labels[:5]                      # substitute your live window here
    stage, conf, pic = predictor.predict_next_stage(recent, return_frame=True)
    print(f"Recent window: {recent}")
    print(f"Predicted next stage = {stage}  (confidence = {conf:.2%})")

    # 5. visualise the *picture* associated to the predicted stage
    plt.imshow(pic, cmap="gray")
    plt.title(f"Predicted Stage {stage}")
    plt.axis("off")
    plt.show()
