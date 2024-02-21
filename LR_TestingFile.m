data = readtable('X_test.csv');

load('logisticmodel.mat')
y_pred = predict(model, data);

% Calculate the accuracy
accuracy2 = sum(y_pred == y_test) / length(y_test);

% Display the accuracy
disp(['Accuracy: ', num2str(accuracy2)])