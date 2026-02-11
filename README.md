Aspect-Based Sentiment Analyzer (ABSA)A deep learning project that performs Aspect-Based Sentiment Analysis. Unlike standard sentiment analysis, 
this model identifies the sentiment specifically for a particular aspect (feature) mentioned in a review.Example: "The battery is great but the screen is dim.


"Aspect: battery ➔ Positive 🟢Aspect: screen ➔ Negative 🔴🚀 FeaturesLSTM-Based Architecture: Uses a Long Short-Term Memory (LSTM) network to capture context between the review and the specific aspect.MLflow Integration: Full experiment tracking, including hyperparameter logging and model versioning.



PyTorch Lightning: Structured and scalable training code.Gradio Web Interface: A user-friendly UI for real-time inference.Hugging Face Ready: Includes scripts optimized for seamless deployment to Hugging Face Spaces.

🛠️ Tech StackDeep Learning: PyTorch, PyTorch LightningNLP: Regex, Custom Tokenization, Word EmbeddingsTracking: 


MLflowUI/Deployment: Gradio, Hugging Face SpacesData Handling: Pandas, Numpy



📁 Project StructurePlaintext



├── app.py                # Deployment script for Hugging Face / Gradio
├── train_model.py        # Local training script (PyTorch Lightning + MLflow)
├── vocab.json            # Exported word-to-ID mapping
├── model_weights.pth     # Trained LSTM weights
├── requirements.txt      # Project dependencies
└── data/                 # Folder for train (2).csv and test (2).csv



⚙️ Installation & Usage1. Clone the RepositoryBashgit clone https://github.com/YOUR_USERNAME/AspectBasedSentimentAnalyzer.git
cd AspectBasedSentimentAnalyzer

2. Install DependenciesBashpip install -r requirements.txt
3. Run Training (Local)Ensure your CSV data paths are updated in train_model.py, then run:Bashpython train_model.py
View logs via MLflow: mlflow ui4. Run the AppBashpython app.py

📊 Model PerformanceThe model uses a Many-to-One LSTM architecture.Hidden Dimension: 512Embedding Dimension: 256Optimizer: Adam ($lr = 3 \times 10^{-4}$)Loss Function: Cross-Entropy LossMetricScoreTrain Accuracy~70% Test Accuracy~68%🌟 Live DemoTry the model directly on Hugging Face 
Spaces:👉 [(https://huggingface.co/spaces/tirtha4542/AspectBasedSentimentAnalyzerApp)]🤝 
ContributingContributions are welcome! If you have ideas for improving the model (e.g., using Transformers/BERT), feel free to open an issue or submit a pull request.
