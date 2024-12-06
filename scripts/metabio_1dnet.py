import tensorflow as tf
from tensorflow.keras import layers, models

# Custom CNN Class
class MetaBioClassifier1D(tf.keras.Model):
    def __init__(self, input_features=110, intermediate_features=1024):
        super(MetaBioClassifier1D, self).__init__()
        
        # Pre-linear layer: Dense + ReLU + BatchNorm + Dropout
        self.pre_linear = models.Sequential([
            layers.Dense(intermediate_features, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3)
        ])
        
        # Convolutional layers: # 3 x (Conv + ReLU) + BatchNorm + Dropout + Pool
        self.conv_layers = models.Sequential([
            layers.Conv1D(16, kernel_size=3, strides=1, padding='same', activation='relu'),

            # layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling1D()
        ])
        
        # Fully connected layers: 2 X (Dense + ReLU) + BatchNorm + Dropout + Output
        self.fc_layers = models.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Output layer
            layers.Dense(3)
        ])

    def call(self, inputs, training=False):
        x = self.pre_linear(inputs, training=training)
        # Reshape for Conv1D input
        x = tf.reshape(x, (-1, 1, x.shape[-1]))  
        x = self.conv_layers(x, training=training)
        x = self.fc_layers(x, training=training)
        return x

# Custom Trainer Class
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer=None, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Using SparseCategoricalCrossentropy as the target is multiclass varible instead of one hot encoded
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.class_weights = class_weights

        # Lists to store training and validation metrics across epochs
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    # Performs one iteration of training
    def train_epoch(self):
        # Metrics for tracking loss and accuracy during training
        train_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Iterate over batches in the training dataset
        for X_batch, y_batch in self.train_loader:
            with tf.GradientTape() as tape:
                # Forward propogation
                logits = self.model(X_batch, training=True)
                loss = self.loss_fn(y_batch, logits)

                # Apply class weights if provided to adjust the loss
                if self.class_weights:
                    weights = tf.gather(self.class_weights, y_batch)
                    loss = tf.reduce_mean(loss * weights)

            # Backward propogation
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # Update loss and accuracy metrics
            train_loss.update_state(loss)
            train_accuracy.update_state(y_batch, logits)

        # Append train metrics to the lists
        self.train_losses.append(train_loss.result().numpy())
        self.train_accuracies.append(train_accuracy.result().numpy())

        return train_loss.result().numpy(), train_accuracy.result().numpy()

    # Performs one iteration of validation.
    def validate_epoch(self):
        # Metrics for tracking validation loss and accuracy
        val_loss = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        all_preds, all_labels = [], []

        for X_batch, y_batch in self.val_loader:
            # Forward propogation without gradient calculation
            logits = self.model(X_batch, training=False)
            loss = self.loss_fn(y_batch, logits)

            # Update loss and accuracy metrics
            val_loss.update_state(loss)
            val_accuracy.update_state(y_batch, logits)

            # Extract predictions and store them
            preds = tf.argmax(logits, axis=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

        # Append validation metrics to the lists
        self.val_losses.append(val_loss.result().numpy())
        self.val_accuracies.append(val_accuracy.result().numpy())

        return val_loss.result().numpy(), val_accuracy.result().numpy(), all_preds, all_labels

    # Main training loop
    def train(self, epochs):
        print("Running Training...")
        all_preds, all_labels = [], []
        
        # Perform training and validation for one epoch
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy, preds, labels = self.validate_epoch()
            all_preds, all_labels = preds, labels
            
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
        return all_preds, all_labels

    # performs prediction on test dataset
    def predict(self, test_loader):
        all_preds, all_labels = [], []

        # Forward propogation without gradient calculation
        for X_batch, y_batch in test_loader:
            logits = self.model(X_batch, training=False)

            # Extract predictions and store them
            preds = tf.argmax(logits, axis=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

        return all_preds, all_labels

    def save_model(self, file_path):
        self.model.save_weights(file_path)

    def load_model(self, file_path):
        self.model.load_weights(file_path)
