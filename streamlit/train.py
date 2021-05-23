from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, average_precision_score, confusion_matrix, f1_score,recall_score, precision_score
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def linearRegr(x_train, x_test, y_train, y_test):
    model = LinearRegression().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    r2 = r2_score(y_test,ypred)
    return model, mae, mse, r2

def svr(x_train, x_test, y_train, y_test):
    model = SVR().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    r2 = r2_score(y_test,ypred)
    return model, mae, mse, r2

def decisionTreeRegr(x_train, x_test, y_train, y_test):
    model = DecisionTreeRegressor().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    r2 = r2_score(y_test,ypred)
    return model, mae, mse, r2

def randomForestRegr(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    r2 = r2_score(y_test,ypred)
    return model, mae, mse, r2

def logisticRegress(x_train, x_test, y_train, y_test):
    model = LogisticRegression().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    acc = accuracy_score(y_test, ypred)
    avp = average_precision_score(y_test, ypred)
    conf = confusion_matrix(y_test, ypred)
    f1 = f1_score(y_test,ypred)
    recall = recall_score(y_test,ypred)
    prec = precision_score(y_test,ypred)
    return model, mae, acc, avp, conf, f1, recall, prec

def kNN(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    acc = accuracy_score(y_test, ypred)
    avp = average_precision_score(y_test, ypred)
    conf = confusion_matrix(y_test, ypred)
    f1 = f1_score(y_test,ypred)
    recall = recall_score(y_test,ypred)
    prec = precision_score(y_test,ypred)
    return model, mae, acc, avp, conf, f1, recall, prec

def svc(x_train, x_test, y_train, y_test):
    model = SVC().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    acc = accuracy_score(y_test, ypred)
    avp = average_precision_score(y_test, ypred)
    conf = confusion_matrix(y_test, ypred)
    f1 = f1_score(y_test,ypred)
    recall = recall_score(y_test,ypred)
    prec = precision_score(y_test,ypred)
    return model, mae, acc, avp, conf, f1, recall, prec

def decisionTreeClass(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    acc = accuracy_score(y_test, ypred)
    avp = average_precision_score(y_test, ypred)
    conf = confusion_matrix(y_test, ypred)
    f1 = f1_score(y_test,ypred)
    recall = recall_score(y_test,ypred)
    prec = precision_score(y_test,ypred)
    return model, mae, acc, avp, conf, f1, recall, prec

def GaussianNaive(x_train, x_test, y_train, y_test):
    model = GaussianNB().fit(x_train, y_train)
    ypred = model.predict(x_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, ypred)
    acc = accuracy_score(y_test, ypred)
    avp = average_precision_score(y_test, ypred)
    conf = confusion_matrix(y_test, ypred)
    f1 = f1_score(y_test,ypred)
    recall = recall_score(y_test,ypred)
    prec = precision_score(y_test,ypred)
    return model, mae, acc, avp, conf, f1, recall, prec



