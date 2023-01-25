from sklearn.ensemble import GradientBoostingClassifier as GBC
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from sklearn.metrics import confusion_matrix, f1_score

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1.0, help="Impact of each estimator on total result")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of iterations")
    parser.add_argument('--subsample', type=float, default=1.0, help="Subsampling rate")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Learning rate:", np.float(args.learning_rate))
    run.log("Number of iterations:", np.int(args.n_estimators))
    run.log("Subsampling rate:", np.int(args.subsample))

    ws = run.experiment.workspace

    dataset = Dataset.get_by_name(ws, name='capstone_dataset')
    ds = dataset.to_pandas_dataframe()

    x = ds.drop(columns='y')

    y = ds['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    model = GBC(learning_rate=args.learning_rate, n_estimators=args.n_estimators, subsample=args.subsample).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    run.log("Accuracy", np.float(accuracy))
    run.log("F1Score", f1_score(y_test, y_pred))
    
    labels = np.union1d(y_pred, y_test).tolist()
    cmtx = confusion_matrix(y_test, y_pred, labels=labels)
    cmtx = {"class_labels": labels,
            "matrix": [[int(y) for y in x] for x in cmtx]}

    run.log_confusion_matrix('Confusion matrix', cmtx)
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/Hyperdrive_capstone.joblib')
    
if __name__ == '__main__':
    main()