% Read the CSV file into a table
data = readtable('final_stress_dataset.csv');

% Display the first few rows of the table
%disp(data(1:5, :))

% Separate the target variable
X = data(:, setdiff(data.Properties.VariableNames, {'stress_level'}));
y = data.stress_level;


rows = size(X, 1);

% Set the random seed for reproducibility
rng('default');

% Generate a random permutation of indices
rand_values = randperm(rows);

% Calculate the number of test samples
test_data = floor(rows * 0.25);

% Create the test set indices
testIndices = rand_values(1:test_data);

% Create the train set indices
trainIndices = rand_values(test_data+1:end);

% Split the data into training and testing sets
X_train = X(trainIndices, :);
X_test = X(testIndices, :);
y_train = y(trainIndices, :);
y_test = y(testIndices, :);

% Create a template for a logistic regression learner
log_reg = templateLinear('Learner', 'logistic');

% Fit the ECOC model to the training data
model = fitcecoc(X_train, y_train, 'Learners', log_reg, 'Coding', 'onevsall');

% Make predictions on new data (e.g., X_test)
y_pred = predict(model, X_test);

% Calculate the accuracy
accuracy2 = sum(y_pred == y_test) / length(y_test);

% Display the accuracy
disp(['Accuracy: ', num2str(accuracy)])


% Standardize Training Data
mu = mean(X_train);
sigma = std(X_train);
Xtrain_std = (X_train - mu) ./ sigma;

% Define hyperparameters
C_values = [ 1, 10];
penalty = {'ridge', 'lasso'};

% Initialize variables to track the best model
logistic_bestModel = [];
Accuracy = 0;
bestParams = struct('C', [], 'penalty', []);

% Loop over the grid
for c = C_values
    for penalty = penalty
        % Template for binary learner
        temp = templateLinear('Regularization', penalty{1}, 'Lambda', 1/c);

        % Train the multi-class model using fitcecoc
        model2 = fitcecoc(Xtrain_std, y_train, 'Learners', temp, 'Coding', 'onevsall');

        % Standardize the validation data using the same parameters
        Xval_std = (X_test - mu) ./ sigma;

        % Predict on the validation set
        y_pred = predict(model2, Xval_std);

        % Calculate accuracy    
        opt_accuracy = sum(y_pred == y_test) / length(y_test);

        % Update best model if current model is better
        if accuracy > opt_accuracy
            logistic_bestModel = model2;
            bestAccuracy = accuracy;
            bestParams.C = c;
            bestParams.penalty = penalty{1};
        end
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

fprintf('Test Accuracy: %.2f%%\n', optimised_accuracy * 100);
fprintf('Best Accuracy: %.2f%%\n', accuracy2 * 100);
disp(bestParams)
% bestModel, bestAccuracy, and bestParams now contain the information about the best model and its parameters
save('logisticmodel.mat',"model")
% Then write the table to a CSV file

%writetable(X_test, 'X_test.csv');