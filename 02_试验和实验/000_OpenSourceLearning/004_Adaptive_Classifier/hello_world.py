from adaptive_classifier import AdaptiveClassifier

# 🎯 Step 1: Initialize with any HuggingFace model
classifier = AdaptiveClassifier("bert-base-uncased")

# 📝 Step 2: Add training examples
texts = ["The product works great!", "Terrible experience", "Neutral about this purchase"]
labels = ["positive", "negative", "neutral"]
classifier.add_examples(texts, labels)

# 🔮 Step 3: Make predictions
predictions = classifier.predict("This is amazing!")
print(predictions)
# Output: [('positive', 0.85), ('neutral', 0.12), ('negative', 0.03)]
# 我的output：[('positive', 0.7175794500256356), ('neutral', 0.1454683679610677), ('negative', 0.13695218201329676)]
