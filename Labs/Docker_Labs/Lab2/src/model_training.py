import joblib
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    joblib.dump(sc, 'scaler.pkl')

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'knn_model.pkl')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    model.save('my_model.keras')
    print('Trained and saved: my_model.keras, scaler.pkl, knn_model.pkl')
