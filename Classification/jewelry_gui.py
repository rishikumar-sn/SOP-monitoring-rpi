from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from jewelry_classifier import (
    DEFAULT_CACHE_FILE,
    DEFAULT_ONNX_MODEL,
    DEFAULT_PROMPT_FILE,
    DEFAULT_TEXT_MODEL_ID,
    JewelryZeroShotClassifier,
    PredictionResult,
)

PREVIEW_SIZE = (360, 360)


class JewelryClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Jewelry Classifier")
        self.root.geometry("1080x760")
        self.root.minsize(960, 680)

        self.classifier: JewelryZeroShotClassifier | None = None
        self.selected_path: Path | None = None
        self.original_preview: ImageTk.PhotoImage | None = None
        self.cropped_preview: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="Loading model and prompts...")
        self.path_var = tk.StringVar(value="No image selected")
        self.result_var = tk.StringVar(value="Prediction will appear here")

        self._build_layout()
        self._set_busy(True, "Loading model and prompts...")
        self.root.after(100, self._load_classifier_async)

    def _build_layout(self) -> None:
        self.root.configure(bg="#f5f0e8")

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="SigLIP2 Jewelry Classifier",
            font=("Segoe UI Semibold", 20),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            outer,
            text="ONNX image encoder on laptop now, with prompt-based zero-shot matching.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w", pady=(2, 14))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 14))

        self.load_button = ttk.Button(
            controls,
            text="Load Image",
            command=self._select_image,
        )
        self.load_button.pack(side="left")

        self.classify_button = ttk.Button(
            controls,
            text="Classify",
            command=self._classify_selected_image,
            state="disabled",
        )
        self.classify_button.pack(side="left", padx=(8, 0))

        path_label = ttk.Label(
            controls,
            textvariable=self.path_var,
            font=("Segoe UI", 10),
        )
        path_label.pack(side="left", padx=(16, 0))

        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)

        preview_frame = ttk.LabelFrame(content, text="Preview", padding=12)
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        preview_grid = ttk.Frame(preview_frame)
        preview_grid.pack(fill="both", expand=True)
        preview_grid.columnconfigure(0, weight=1)
        preview_grid.columnconfigure(1, weight=1)

        original_title = ttk.Label(
            preview_grid,
            text="Original",
            font=("Segoe UI Semibold", 11),
        )
        original_title.grid(row=0, column=0, sticky="w", pady=(0, 6))

        cropped_title = ttk.Label(
            preview_grid,
            text="Foreground Crop",
            font=("Segoe UI Semibold", 11),
        )
        cropped_title.grid(row=0, column=1, sticky="w", pady=(0, 6), padx=(10, 0))

        self.original_label = ttk.Label(
            preview_grid,
            text="No image",
            anchor="center",
            relief="solid",
            width=42,
        )
        self.original_label.grid(row=1, column=0, sticky="nsew")

        self.cropped_label = ttk.Label(
            preview_grid,
            text="No crop yet",
            anchor="center",
            relief="solid",
            width=42,
        )
        self.cropped_label.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        result_frame = ttk.LabelFrame(content, text="Result", padding=12)
        result_frame.pack(side="right", fill="both", expand=False)

        result_title = ttk.Label(
            result_frame,
            textvariable=self.result_var,
            font=("Segoe UI Semibold", 18),
            wraplength=300,
            justify="left",
        )
        result_title.pack(anchor="w", pady=(0, 12))

        columns = ("label", "confidence", "similarity")
        self.score_table = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=8,
        )
        self.score_table.heading("label", text="Class")
        self.score_table.heading("confidence", text="Confidence")
        self.score_table.heading("similarity", text="Similarity")
        self.score_table.column("label", width=150, anchor="w")
        self.score_table.column("confidence", width=120, anchor="center")
        self.score_table.column("similarity", width=120, anchor="center")
        self.score_table.pack(fill="both", expand=True)

        status_bar = ttk.Label(
            outer,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 9),
        )
        status_bar.pack(fill="x", pady=(14, 0))

    def _set_busy(self, busy: bool, message: str) -> None:
        self.status_var.set(message)
        load_state = "disabled" if busy else "normal"
        classify_state = "disabled"
        if not busy and self.selected_path is not None and self.classifier is not None:
            classify_state = "normal"
        self.load_button.config(state=load_state)
        self.classify_button.config(state=classify_state)

    def _run_in_background(self, work, on_success, failure_message: str) -> None:
        def worker() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                self.root.after(
                    0,
                    lambda: self._handle_background_error(failure_message, exc),
                )
                return
            self.root.after(0, lambda: on_success(result))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_background_error(self, message: str, exc: Exception) -> None:
        self._set_busy(False, f"{message}: {exc}")
        messagebox.showerror("Jewelry Classifier", f"{message}\n\n{exc}")

    def _load_classifier_async(self) -> None:
        def work() -> JewelryZeroShotClassifier:
            return JewelryZeroShotClassifier(
                onnx_model_path=DEFAULT_ONNX_MODEL,
                prompt_path=DEFAULT_PROMPT_FILE,
                text_model_id=DEFAULT_TEXT_MODEL_ID,
                embedding_cache_path=DEFAULT_CACHE_FILE,
            )

        def on_success(classifier: JewelryZeroShotClassifier) -> None:
            self.classifier = classifier
            self._set_busy(False, "Model ready. Load an image to classify.")

        self._run_in_background(
            work,
            on_success,
            "Unable to load the ONNX model or text prompts",
        )

    def _select_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select a jewelry image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.selected_path = Path(file_path)
        self.path_var.set(str(self.selected_path))
        self.result_var.set("Image loaded. Click Classify.")
        self._show_single_preview(self.selected_path)
        if self.classifier is not None:
            self.classify_button.config(state="normal")
        self.status_var.set("Image ready.")
        self._clear_scores()

    def _show_single_preview(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        preview = self._make_preview(image)
        self.original_preview = preview
        self.original_label.config(image=preview, text="")
        self.cropped_label.config(image="", text="Crop will appear after classification")
        self.cropped_preview = None

    def _make_preview(self, image: Image.Image) -> ImageTk.PhotoImage:
        preview = image.copy()
        preview.thumbnail(PREVIEW_SIZE)
        return ImageTk.PhotoImage(preview)

    def _classify_selected_image(self) -> None:
        if self.classifier is None or self.selected_path is None:
            return

        self._set_busy(True, "Running ONNX image encoder and matching prompts...")

        def work() -> PredictionResult:
            return self.classifier.classify_path(self.selected_path)

        def on_success(prediction: PredictionResult) -> None:
            self._render_prediction(prediction)
            self._set_busy(False, f"Finished. Predicted {prediction.label}.")

        self._run_in_background(
            work,
            on_success,
            "Unable to classify the selected image",
        )

    def _render_prediction(self, prediction: PredictionResult) -> None:
        self.result_var.set(
            f"Predicted class: {prediction.label}\n"
            f"Confidence: {prediction.confidence:.2%}"
        )
        self.original_preview = self._make_preview(prediction.original_image)
        self.original_label.config(image=self.original_preview, text="")

        self.cropped_preview = self._make_preview(prediction.cropped_image)
        self.cropped_label.config(image=self.cropped_preview, text="")

        self._clear_scores()
        for score in prediction.scores:
            self.score_table.insert(
                "",
                "end",
                values=(
                    score.label,
                    f"{score.confidence:.2%}",
                    f"{score.similarity:.4f}",
                ),
            )

    def _clear_scores(self) -> None:
        for item in self.score_table.get_children():
            self.score_table.delete(item)


def main() -> None:
    root = tk.Tk()
    app = JewelryClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
