# Python script to compare classification algorithms

from matplotlib.colors import Normalize
# create colour normalizing function for heatmap to be used in GridSearch
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def classifier_comparison(features, y_features):
  #split data into train and test
  X_train, X_test, y_train, y_test = train_test_split(features, y_features, random_state = 0)

  # run all optimising functions to find optimal parameters
  opitmal_SVM_ovo = my_optimal_SVM_ovo(features, y_features)
  opitmal_SVM_ovr = my_optimal_SVM_ovr(features, y_features)
  optimal_KNN = my_optimal_KNN(features, y_features)
  optimal_FOREST = my_optimal_FOREST(features, y_features)

  # using optimal parameters output
  # - confusion matrix for each classifier
  # - table with classifier name, specificity, sensitivity, accuracy


def my_optimal_SVM_ovo(features, y_features):
    # function to obtain optimal hyper-parameters using gridSearch for RBF SVM
    clf_rbf = svm.SVC(kernel ='rbf', decision_function_shape='ovo')
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(clf_rbf, param_grid=param_grid, cv=cv)
    grid.fit(features, y_features)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))
    # Plotting heatmap
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

    return grid.best_params_


def my_optimal_SVM_ovr(features, y_features):
    # function to obtain optimal hyper-parameters using gridSearch for RBF SVM
    clf_rbf = svm.SVC(kernel ='rbf', decision_function_shape='ovr')
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(clf_rbf, param_grid=param_grid, cv=cv)
    grid.fit(features, y_features)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))
    # Plotting heatmap
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

    return grid.best_params_


def my_optimal_KNN(features, y_features):
    # function to obtain optimal number of neighbours for NN algorithm
    neighbours_range = [1, 10, 20, 30, 40, 50, 60 ,70 ,80, 90, 100, 500, 1000]
    param_grid = dict(n_neighbors = neighbours_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    neigh = KNeighborsClassifier()
    grid = GridSearchCV(neigh, param_grid=param_grid, cv=cv)
    grid.fit(features, y_features)

    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    # plot accuracy of classifier for each neighbours_range
    scores = grid.cv_results_['mean_test_score']
    plt.plot(neighbours_range, scores)
    plt.xlabel('number of neighbours')
    plt.ylabel('validation accuracy')
    plt.show()

    return grid.best_params_

def my_optimal_FOREST(features, y_features):
  # function to obtain optimal hyper-parameters using gridSearch for Random Forest ensemble
  max_depth_range = [1,10,20,30,40,50,60,70,80,90,100]
  n_estimators = [1,5,10,15,20,25,30,35,40,45,50]
  param_grid = dict(max_depth_range = max_depth_range, n_estimators = n_estimators)
  cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
  clf_forest = RandomForestClassifier()
  grid = GridSearchCV(clf_forest, param_grid = param_grid, cv=cv)
  grid.fit(features, y_features)
  print("The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_))

    scores = grid.cv_results_['mean_test_score'].reshape(len(max_depth_range),
                                                     len(n_estimators))
    # Plotting heatmap
    plt.xlabel('number of estimators')
    plt.ylabel('maximum depth range')
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

    return grid.best_params_
