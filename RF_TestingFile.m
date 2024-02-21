data = readtable('X_test.csv');

load('randomforestmodel.mat')
y_pred = predict(rfModel, data);

% Calculate the accuracy
accuracy2 = sum(y_pred == y_test) / length(y_test);

% Display the accuracy
disp(['Accuracy: ', num2str(accuracy2)])