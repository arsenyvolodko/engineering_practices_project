from math import floor, ceil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data():
    train = pd.read_csv(train_file_name)
    test = pd.read_csv(test_file_name)
    res = pd.concat([train, test], axis=0)
    return res


def drop_features(df):
    df.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1, inplace=True)


def encode(df, cat_features):
    le = LabelEncoder()
    for feature in cat_features:
        df[feature] = le.fit_transform(df[feature])
    return df


def plot_features_distribution(df, rational_features, discrete_features):
    discrete_features.remove('Embarked')
    features = rational_features + discrete_features
    sns.set(style="whitegrid")
    for feature in features:
        plt.figure(figsize=(8, 6))
        if feature in discrete_features:
            sns.countplot(x=feature, data=df, palette='viridis', hue='Survived')
        else:
            sns.histplot(df[feature].dropna(), kde=True, color='skyblue', bins=30)
        title = f'Distribution of {feature.capitalize()}'
        plt.title(title)
        plt.savefig(IMAGES_DIR + '/' + title + '.png')


def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(f'{IMAGES_DIR}/Correlation Heatmap.png')
    plt.show()


def get_mean_val(df, mask, feature):
    return df[mask][feature].dropna().mean()


def write_average_data(df, mask, feature, title, file_name):
    with open(file_name, 'a') as f:
        f.write(title + '\n')
        if feature == 'Pclass':
            f.write(f'Возраст: {int(get_mean_val(df, mask, "Age"))}\n')
        else:
            f.write(f"Класс: {df[mask]['Pclass'].mode().iloc[0]}\n")
        f.write(f"Пол: {df[mask]['Sex'].mode().iloc[0]}\n")
        f.write(f'Стоимость билета: {round(get_mean_val(df, mask, "Fare"), 2)}\n')
        parch = get_mean_val(df, mask, "Parch")
        f.write(f'Количество родителей/детей на борту: от {floor(parch)} до {ceil(parch)}\n')
        sibsp = get_mean_val(df, mask, "SibSp")
        f.write(f'Количество братьев/сестер на борту: от {floor(sibsp)} до {ceil(sibsp)}\n')
        f.write(f'Шанс выжить: {round(get_mean_val(df, mask, "Survived"), 2)}\n')
        emark = df[mask]['Embarked'].mode().iloc[0]
        f.write(f'Порт посадки: {emark}\n')
        f.write('\n')


def get_mean_vals(df, feature):
    df = df.copy().dropna()
    if feature == 'Pclass':
        pclass_values = sorted(df[feature].unique())
        for pclass in pclass_values:
            mask = df[feature] == pclass
            title = f'Средний портрет для пассажира {pclass} класса:'
            write_average_data(df, mask, feature, title, passangers_stats_file_name)
    else:
        age_ranges = [(0, 14), (14, 21), (21, 35), (35, 55), (55, df[feature].max() + 1)]
        for age_range in age_ranges:
            mask = (df['Age'] >= age_range[0]) & (df[feature] < age_range[1])
            title = f'Средний портрет для пассажира в возрасте от {int(age_range[0])} до {int(age_range[1])} лет:'
            write_average_data(df, mask, feature, title, passangers_stats_file_name)


def prepare_data(df, features):
    features.remove('Survived')
    df.dropna(inplace=True)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def main():
    titanic_data = load_data()
    drop_features(titanic_data)
    get_mean_vals(titanic_data, 'Pclass')
    get_mean_vals(titanic_data, 'Age')
    rational_features, discrete_features = ['Age', 'Fare'], ['Parch', 'SibSp', 'Survived', 'Pclass', 'Sex', 'Embarked']
    all_features = rational_features + discrete_features
    plot_features_distribution(titanic_data, rational_features, discrete_features)
    cat_features = ['Sex', 'Embarked']
    titanic_data = encode(titanic_data, cat_features)
    plot_correlation_heatmap(titanic_data)
    processed_data = prepare_data(titanic_data, all_features)
    processed_data.to_csv(output_file_name, index=False)


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: python3 preprocess_data.py input_train_data_file input_test_data_file output_processed_data_file output_pas_info_file")
        sys.exit(1)

    IMAGES_DIR = './../results/images'

    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    passangers_stats_file_name = sys.argv[4]
    main()
