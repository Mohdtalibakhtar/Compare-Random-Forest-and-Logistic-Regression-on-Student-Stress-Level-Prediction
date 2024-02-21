% Read the CSV file into a table
data = readtable('final_stress_dataset.csv');

% Display the first few rows of the table
%disp(data(1:5, :))

% Separate the target variable
X = data(:, setdiff(data.Properties.VariableNames, {'stress_level'}));
y = data.stress_level;


rows = size(X, 1);

% Set the random seed for reproducibility
rng(42, 'twister');

% Generate a random permutation of indices
indices = randperm(rows);

% Calculate the number of test samples
test_data = floor(rows * 0.25);

% Create the test set indices
testIndices = indices(1:test_data);

% Create the train set indices
trainIndices = indices(test_data+1:end);

% Split the data into training and testing sets
X_train = X(trainIndices, :);
X_test = X(testIndices, :);
y_train = y(trainIndices, :);
y_test = y(testIndices, :);

% Train the Random Forest model
numTrees = 100; % Number of trees in the ensemble
rfModel = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', numTrees, ...
                       'Learners', 'Tree', 'CategoricalPredictors', 'all');

% Predict the stress levels for the test set
predictions = predict(rfModel, X_test);

% Calculate the accuracy
accuracy2 = sum(predictions == y_test) / numel(y_test);



numTreesGrid = [50, 100, 200, 300]; % Number of trees
% Initialize vector to store accuracy for each number of trees
accuracyResults = zeros(1, length(numTreesGrid));
bestAccuracy = 0;
bestModel = [];
bestNumTrees = 0;
% Grid Search with accuracy storage
for i = 1:length(numTreesGrid)
    % Train the model
    model = fitcensemble(X_train, y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', numTreesGrid(i));
    % Make predictions and evaluate accuracy
    y_pred = predict(model, X_test);
    accuracy = sum(y_pred == y_test) / numel(y_test);
    % % Store accuracy in vector
    accuracyResults(i) = accuracy;
    % Update best model if current model is better
    if accuracy > bestAccuracy
        bestAccuracy = accuracy;
        bestModel = model;
        bestNumTrees = numTreesGrid(i);
    end
end

% Creating the confusion matrix
Conf_mat = confusionmat(y_test, y_pred);

% Displaying the confusion matrix
disp(Conf_mat);

% Visualizing the confusion matrix
confusionchart(y_test, y_pred);

numClasses = size(C, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);

for i = 1:numClasses
    TP = Conf_mat(i, i);
    FP = sum(Conf_mat(:, i)) - TP;
    FN = sum(Conf_mat(i, :)) - TP;
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
end

% Average Precision and Recall
precision_mean = mean(precision);
recall_mean = mean(recall);

% Macro F1 Score
f1score = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean);
disp(f1score);
disp(recall_mean);
disp(precision_mean);


% Save X_test to a CSV file
writetable(X_test, 'X_test.csv');

fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Best Accuracy: %.2f%%\n', accuracy2 * 100);
save('randomforestmodel.mat',"rfModel")