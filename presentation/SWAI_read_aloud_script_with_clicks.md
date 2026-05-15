# SWAI Read-Aloud Presentation Script With Click Cues

Use this file as the live speaking script. The spoken parts are written in English.

Do not read the bracketed cue lines aloud. They tell the presenter when to change slides.

Presentation structure:

- Presenter A: Slides 1-5, Machine Learning
- Presenter B: Slides 6-10, Deep Learning
- Presenter C: Slides 11-15, LLM and RAG
- Presenter D: Slides 16-20, System, Applications, and Future Vision
- Total duration: about 20 minutes
- Each presenter: about 5 minutes

---

# Presenter A - Machine Learning

## Slide 1 - SWAI: AI-Assisted Exoplanet Classification

[START ON SLIDE 1. PRESENTER A BEGINS HERE.]

Good morning everyone. Today we are presenting SWAI, which stands for Silent Watcher AI. Our project is an educational AI prototype for exoplanet candidate classification.

An exoplanet is a planet outside our solar system. In real astronomy, detecting exoplanets is difficult because the signal can be very small, noisy, and easy to confuse with other events. So our project does not claim to replace scientific validation. Instead, SWAI shows how an AI workflow can help screen possible candidates and explain the result to users.

The system has four main parts. First, we prepare exoplanet-style tabular data. Second, we train classification models. Third, we provide a Streamlit web app for manual input and CSV upload. Fourth, we add a science explainer so users can understand terms like transit depth, false positives, and confidence.

In my section, I will explain the machine learning part: the data, the features, the Random Forest model, and the results.

[CLICK TO SLIDE 2.]

## Slide 2 - From Exoplanet Records to Supervised Learning

The learning task starts from exoplanet candidate records. In our project code, the data preparation script fetches Kepler-style and TESS-style records, then normalizes them into one local table.

After filtering and cleaning, our local dataset contains 45,973 rows. Each row has a label from three possible classes. The first class is CONFIRMED, which means the source label treats the object as a validated planet. The second class is CANDIDATE, which means the signal may be a planet but still needs more evidence. The third class is FALSE POSITIVE, which means the signal looks planet-like but is probably caused by something else, such as an eclipsing binary star, background contamination, or instrument noise.

This becomes a supervised learning problem because every training example has input features and a known label. The model learns patterns from these labeled examples, then predicts the label for new observations.

One important limitation is that we are not using raw telescope light curves here. We are using a compact tabular representation, which is easier to train and explain, but also less scientifically complete.

[CLICK TO SLIDE 3.]

## Slide 3 - Four Features Summarize the Transit Signal

The model uses four numeric features. These features are simple, interpretable, and connected to the transit method of exoplanet detection.

The first feature is orbital period. It describes how long the object takes to complete one orbit around the star. If a brightness dip repeats every fixed number of days, that pattern may suggest an orbiting object.

The second feature is transit duration. This measures how long the brightness dip lasts. A real transit has a time structure: the object starts crossing the star, blocks some light, and then exits.

The third feature is transit depth. This is the drop in brightness during the transit, often measured in parts per million. A deeper transit may suggest a larger object, but it can also come from a false positive, so we need to interpret it carefully.

The fourth feature is planet radius, measured in Earth radii. It helps distinguish smaller planet-like candidates from very large or suspicious objects.

These four features are useful because students can understand them, and the web app can expose them directly through sliders.

[CLICK TO SLIDE 4.]

## Slide 4 - Random Forest as the Main ML Baseline

Our main machine learning model is a Random Forest classifier. A Random Forest is an ensemble method, which means it does not depend on only one decision tree. Instead, it trains many trees and combines their decisions.

This is a good choice for our project because Random Forests usually work well on tabular data. They can model nonlinear relationships, they are more robust than a single tree, and they give a strong baseline before we try more complex models.

In our implementation, the Random Forest uses 300 trees. The pipeline also includes median imputation for missing values and balanced class weights to reduce the effect of label imbalance. The data is split with stratification, so the class distribution is preserved in both training and testing.

After training, the model is saved as a joblib artifact. The Streamlit app loads this saved model and calls predict probability. The class with the highest probability becomes the predicted label, and the probability is shown as confidence or trust score.

[CLICK TO SLIDE 5.]

## Slide 5 - Model Results: Strong Baseline, Not Scientific Proof

The Random Forest performs well for a course-level prototype. On the hold-out test set, it achieves about 87.2 percent accuracy and about 0.854 macro-F1.

Accuracy tells us the overall percentage of correct predictions. Macro-F1 is also important because it averages performance across the three classes. This matters because the classes are not equally easy. Confirmed planets are easier to classify, while candidate signals are more ambiguous.

The confusion matrix shows where the model makes mistakes. Some candidate examples may look like confirmed planets. Others may look like false positives. This is scientifically reasonable because candidate objects are, by definition, not fully resolved yet.

So the key message is: model confidence is not the same as scientific confirmation. If the app says a prediction has high confidence, that means the model sees a pattern similar to examples in the training data. It does not mean astronomers have confirmed a planet.

That concludes the machine learning section. Next, Speaker B will explain the deep learning module and why richer light-curve data matters.

[CLICK TO SLIDE 6. SPEAKER B STARTS.]

---

# Presenter B - Deep Learning

## Slide 6 - Why Deep Learning Matters

[START ON SLIDE 6. PRESENTER B BEGINS HERE.]

Thank you. I will now explain the deep learning part of SWAI.

Deep learning matters because exoplanet detection is fundamentally about recognizing subtle patterns in signals. The transit method looks for small, repeated drops in a star's brightness. In a full scientific system, the best input would be the light curve itself, which is a time series of brightness measurements.

However, our current deep learning implementation is intentionally smaller. It does not yet train on raw or folded light curves. Instead, it uses the same four tabular features as the Random Forest: orbital period, transit duration, transit depth, and planet radius.

So we should describe the current neural model as a deep learning MVP. It is useful because it lets us compare a neural approach with the Random Forest inside the same dataset and app. But the real future upgrade is not just using a bigger network. The real upgrade is using richer input data, especially light curves.

[CLICK TO SLIDE 7.]

## Slide 7 - Deep Learning MVP: MLPClassifier

The implemented neural model is an MLPClassifier. MLP means multilayer perceptron. It is a feedforward neural network, so information moves from the input layer, through hidden layers, and finally to the output classes.

The input layer has four values, matching our four transit-related features. The first hidden layer has 64 units, and the second hidden layer has 32 units. The activation function is ReLU, which helps the network learn nonlinear relationships. The output layer predicts three classes: CONFIRMED, CANDIDATE, and FALSE POSITIVE.

The model uses the Adam optimizer. During training, the optimizer adjusts the internal weights of the network to reduce classification error.

This architecture is lightweight enough to train locally and simple enough to explain clearly. It gives the project a working neural-network component without making the whole system too complex.

The trained model is saved as a deep model artifact, so the app can load it and compare it with the machine learning baseline.

[CLICK TO SLIDE 8.]

## Slide 8 - Preprocessing Makes the Neural Model More Stable

For neural networks, preprocessing is very important. In our dataset, some numeric values can be extremely large. For example, orbital period, transit depth, and planet radius can contain outliers. If we train directly on extreme values, the neural model can become unstable or learn poorly.

To handle this, the deep learning pipeline clips features to the same practical ranges used by the app sliders. Orbital period is capped at 1000 days. Transit duration is capped at 30 hours. Transit depth is capped at 20,000 parts per million. Planet radius is capped at 40 Earth radii.

This is an engineering decision. It makes the training distribution more consistent with what users can enter in the web app.

After clipping, the pipeline applies imputation and StandardScaler normalization. Scaling is especially important for neural networks because the four features have very different numeric ranges. If one feature has much larger values than the others, optimization becomes harder.

So this slide is not only about model design. It is also about making the model reliable enough for an interactive app.

[CLICK TO SLIDE 9.]

## Slide 9 - MLP Works, But RF Is Stronger Today

The MLP model works, but it does not outperform the Random Forest.

According to the saved metrics, the MLP reaches about 71.5 percent accuracy and about 0.671 macro-F1. The Random Forest reaches about 87.2 percent accuracy and about 0.854 macro-F1.

This comparison is important because it teaches a realistic lesson: deep learning is not automatically better. If the input is only four tabular features, a tree-based ensemble can be stronger than a small neural network.

The MLP especially struggles with the candidate class. This makes sense because candidates are ambiguous. They are not clearly confirmed planets, but they are not clearly false positives either. To separate them better, the model needs richer evidence.

So the honest conclusion is not that our deep learning model is the best model. The conclusion is that we implemented a neural MVP, measured it against a strong baseline, and learned what needs to improve next.

[CLICK TO SLIDE 10.]

## Slide 10 - Future Direction: Learning Directly from Light Curves

The most meaningful deep learning upgrade is to change the input representation. Instead of learning from only four summary features, the system should learn directly from light curves.

A light curve shows how a star's brightness changes over time. If a planet crosses in front of the star, the brightness may drop slightly and then return to normal. A convolutional neural network could learn local transit-shaped patterns. A Transformer could model longer-range temporal structure, repeated events, and relationships across the sequence.

This future version still needs to be compared against the Random Forest baseline. A more advanced model is only useful if it improves accuracy, calibration, interpretability, or educational value.

Another improvement would be explainability for the deep model. For example, the system could highlight which part of the light curve influenced the prediction. That would make deep learning more trustworthy and easier to teach.

That concludes my deep learning section. Next, Speaker C will explain the LLM and RAG part, especially how SWAI explains predictions responsibly.

[CLICK TO SLIDE 11. SPEAKER C STARTS.]

---

# Presenter C - LLM and RAG

## Slide 11 - A Prediction Is Not Enough

[START ON SLIDE 11. PRESENTER C BEGINS HERE.]

Thank you. I will now explain the explanation layer, including the current local explainer and the future RAG direction.

A model can output a class label and a confidence score, but that alone is not enough for users to understand the result. If the app says CANDIDATE or FALSE POSITIVE, a beginner may not know what that means scientifically. If the app says confidence is 82 percent, the user may think that the planet is 82 percent confirmed, which is not correct.

This is why SWAI includes an explanation layer. The classifier is responsible for prediction. The explainer is responsible for communication.

The explainer helps users understand ideas like transit depth, false positives, model confidence, machine learning, deep learning, reinforcement learning, and the possible role of LLMs.

The important design boundary is that the language layer does not decide the class. It does not replace the classifier. It helps the user interpret the model output responsibly.

[CLICK TO SLIDE 12.]

## Slide 12 - Current MVP: Local Grounded Science Explainer

The current explainer is not a full external LLM system. It is a local, grounded, rule-based science explainer implemented in the project code.

The app takes the user's question, normalizes the text, and checks for topic keywords. For example, if the question mentions transit depth, brightness, or ppm, the explainer returns an answer about how transit depth represents the drop in starlight during a transit. If the question mentions false positives, eclipsing binaries, or noise, it returns an answer about why a signal may look planet-like but not actually be a planet.

The advantage of this MVP is safety and reliability. It does not require an external API key. It does not hallucinate like a general chatbot might. It gives deterministic answers that are grounded in the topics we expect users to ask.

The limitation is flexibility. It cannot answer every possible astronomy question, and it does not retrieve external documents dynamically yet.

[CLICK TO SLIDE 13.]

## Slide 13 - From Model Output to Human Explanation

One useful design detail is that the explainer can use the current prediction context.

The app stores the latest predicted class, the confidence score, and the four input values. Then it formats this information into readable context. For example, the context may say: the current prediction is CONFIRMED, trust is 82 percent, orbital period is 17 days, transit duration is 9 hours, transit depth is 1000 parts per million, and planet radius is 7 Earth radii.

When the user asks a question, the explainer can include this context in the answer. This makes the response specific to the current prediction instead of generic. A student can ask, "What does this result mean?" and the system can explain both the class and the input values.

This is also the bridge toward a future LLM system. A future LLM should not answer in isolation. It should receive structured context from the model and the app state.

[CLICK TO SLIDE 14.]

## Slide 14 - Future RAG: Grounded Scientific Answers

RAG stands for retrieval-augmented generation. The idea is that before the language model answers, the system retrieves relevant information from trusted sources. Then the LLM uses those retrieved passages to generate a grounded answer.

For SWAI, a future RAG system could retrieve from NASA documentation, Exoplanet Archive metadata, mission glossaries, project documentation, or curated paper summaries.

For example, if a user asks about false positives, the retriever could find source material about eclipsing binaries, instrumental artifacts, or validation methods. If a user asks about transit depth, it could retrieve a trusted explanation and connect it to the user's input values.

The answer should include citations or at least source labels. It should also explain uncertainty. For example, it should say that SWAI's prediction is a model estimate, not final scientific confirmation.

The key point is that this RAG architecture is future work. We should not claim that dynamic retrieval is already implemented in the current system.

[CLICK TO SLIDE 15.]

## Slide 15 - Explaining, Not Replacing Science

The language layer must be designed carefully because scientific topics can be easily misunderstood.

SWAI should not say, "we discovered a planet," just because the model predicts CONFIRMED or CANDIDATE. The system should say that the model found a pattern similar to examples in the training data.

There are three different concepts. First, prediction: the model estimates the most likely class. Second, explanation: the language layer explains what the result means. Third, scientific confirmation: astronomers need additional evidence, follow-up observations, and expert validation before making a final claim.

This boundary also applies to future RAG. Even if a future LLM retrieves trusted sources, it should still not replace scientific review. Its role should be to explain, summarize, cite, and guide users toward responsible interpretation.

So SWAI is strongest as an educational AI system. It teaches users how AI can support scientific exploration, while also teaching the limits of AI.

That concludes the LLM and RAG section. Next, Speaker D will connect all modules into the full system architecture and future roadmap.

[CLICK TO SLIDE 16. SPEAKER D STARTS.]

---

# Presenter D - System, Applications, and Future Vision

## Slide 16 - Integrated System Architecture

[START ON SLIDE 16. PRESENTER D BEGINS HERE.]

Thank you. I will now explain SWAI as a complete integrated system.

The project is not just one model file. It has several connected layers. The data layer fetches and stores exoplanet candidate records. The training layer builds the Random Forest model and the MLP model. The artifact layer stores trained models and metrics. The app layer uses Streamlit to provide manual sliders, CSV upload, probability bars, metrics display, and visual media. The explanation layer answers science questions using the local grounded explainer.

This layered design is useful because each part has a clear responsibility. Data preparation is separate from model training. Model training is separate from inference. Inference is separate from explanation. This separation makes the system easier to test, explain, and extend.

Future modules, such as RAG and reinforcement learning, should be treated as roadmap items. They can be shown as future layers, but we should not present them as fully implemented features.

[CLICK TO SLIDE 17.]

## Slide 17 - Interactive Prediction Workflow

The user interaction is designed as a learning loop.

A user can enter values manually using sliders. These values represent orbital period, transit duration, transit depth, and planet radius. When the user runs prediction, the app builds a table with these features and sends it into the model.

The app then displays the predicted class, the trust score, and the probability distribution across the three classes. If the deep learning model is available, the app can also show the MLP result for comparison. This helps users see that different models can behave differently.

There is also a CSV workflow. Users can upload a CSV with the required columns, and the app produces predictions for multiple rows. This is useful when users want to test many observations at once instead of adjusting sliders manually.

Finally, users can ask the science explainer questions about the result. This closes the loop: input, prediction, visualization, explanation, and better understanding.

[CLICK TO SLIDE 18.]

## Slide 18 - Application Scenarios: Education and Open Science

The strongest application scenario for SWAI is education.

In a classroom, students can change the input features and immediately see how the prediction changes. This helps them connect astronomy concepts with machine learning behavior. For example, they can observe how transit depth and planet radius may influence the output.

The second scenario is public science exploration. Space enthusiasts or beginner learners can interact with exoplanet-style data concepts without needing to read raw scientific tables. The interface makes the topic more accessible.

The third scenario is a prototype for AI-assisted candidate screening. In a real research setting, a system like this would not make final decisions. But it could help prioritize which objects deserve closer inspection, especially when combined with better validation, calibrated uncertainty, and trusted explanations.

The key point is that SWAI connects astronomy, data science, and responsible AI. Users do not just see a prediction. They also learn why prediction is uncertain.

[CLICK TO SLIDE 19.]

## Slide 19 - Future Roadmap

The roadmap has three stages.

The current stage is the MVP. It includes a Random Forest classifier, an MLP neural model, a local science explainer, and a Streamlit interface.

The next stage should improve reliability. We can add feature importance so users can see which features influence predictions. We can improve probability calibration so confidence scores become more meaningful. We can also add better validation and monitoring, especially if the model is tested on new data.

The deeper technical stage is richer AI. On the modeling side, this means using light curves with CNNs or Transformers. On the explanation side, this means adding RAG over trusted sources, so answers can include citations. On the decision-support side, reinforcement learning could eventually help prioritize follow-up observations under limited telescope time.

However, a roadmap is not the same as implementation. The project is strongest when it clearly separates what exists now, what can be improved next, and what belongs to long-term research.

[CLICK TO SLIDE 20.]

## Slide 20 - Open Space Data Becomes Understandable AI

To conclude, SWAI is best understood as a layered educational AI system.

The machine learning layer turns public exoplanet-style records into a supervised classification task. The Random Forest model is the strongest implemented baseline and gives the app reliable predictions for the current dataset.

The deep learning layer gives the project a neural MVP and a future direction. The current MLP does not beat the Random Forest, but it teaches an important lesson: deep learning needs the right input representation. Future light-curve models could make this part much stronger.

The language layer makes the system understandable. The current explainer is local and grounded. Future RAG could connect the app to trusted scientific sources, but it should explain rather than replace science.

The system layer brings everything together in an interactive Streamlit app with manual prediction, CSV batch prediction, probability visualization, media, and explanations.

Our final message is this: SWAI does not claim to discover planets by itself. It shows how open scientific data, AI classification, and explainable interaction can help people learn how exoplanet discovery works.

Thank you.

[END OF PRESENTATION. DO NOT CLICK AGAIN UNLESS THERE IS A Q&A OR BACKUP SLIDE.]

