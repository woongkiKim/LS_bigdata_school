import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataCleansing:

    def __init__(self):

        ## 요금 범위
        self.fare_ranges = [  0, 7.2292, 7.75, 7.8958,
                            8.05, 10.5, 13, 15.85,
                            24, 26.55, 33.30769231, 55.9,
                            83.1583, 512.3292]
        ## 나이 범위
        self.age_ranges = [ 0.42, 16, 20, 22, 25, 26, 30, 34, 40,
                            47, 80]
        ## 범주형 변수
        self.cat_cols = ['Sex', 'Embarked', 'Deck', 'Family_Size_Class', 'Title', 'Pclass']

        # 원래 있는 칼럼 목록
        self.columns_to_add = ['Age', 'Fare', 'Ticket_count', 'Is_Married', 'Sex_female', 'Sex_male',
                        'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Deck_ABC', 'Deck_DE',
                        'Deck_FG', 'Deck_Z', 'Family_Size_Class_Alone',
                        'Family_Size_Class_Large', 'Family_Size_Class_Medium',
                        'Family_Size_Class_Small', 'Title_Master', 'Title_Mr', 'Title_Mrs',
                        'Title_Other', 'Pclass_1', 'Pclass_2', 'Pclass_3']
        
        # 학습데이터 Ticket 가중치
        self.ticket_weight = pd.read_csv('./data/titanic/ticket_weight.csv')
        self.ticket_weight = self.ticket_weight.set_index('Ticket')['Weight'].to_dict()

        # 학습데이터 Age 정규화 통계치
        self.age_mean = 4.32210999
        self.age_std = 2.87971488

        # 학습데이터 Fare 정규화 통계치
        self.fare_mean = 5.91133558
        self.fare_std = 3.78049333

        # 학습데이터 Ticket_count 정규화 통계치
        self.ticket_count_mean = 1.78787879
        self.ticket_count_std = 1.36114174

        
    def preprocess(self, df):

        df['Fare'] = df['Fare'].astype(float)
        df['Age'] = df['Age'].astype(int)
        df['SibSp'] = df['SibSp'].astype(int)
        df['Parch'] = df['Parch'].astype(int)

        df['Cabin'].fillna('Z', inplace=True)

        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')
        df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')
        df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')
        df['Deck'] = df['Deck'].replace('T', 'Z')
        df.loc[~df['Deck'].isin(['ABC', 'DE', 'FG']), 'Deck'] = 'Z'

        df['Fare'] = pd.cut(df['Fare'], bins=self.fare_ranges, include_lowest=True)
        df['Fare'] = df['Fare'].astype('category')
        df['Fare'] = df['Fare'].cat.codes

        
        df['Age'] = pd.cut(df['Age'], bins=self.age_ranges, include_lowest=True)
        df['Age'] = df['Age'].astype('category')
        df['Age'] = df['Age'].cat.codes

        
        df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
        df['Family_Size_Class'] = df['Family_Size'].apply(
            lambda x: 'Alone' if x == 1 else 
            ('Small' if x < 5 else 
            ('Medium' if x < 7 else 'Large'))
        )

        # Ticket 가중치
        df['Ticket'] = df['Ticket'].map(self.ticket_weight)
        df['Ticket'].fillna(1, inplace=True)
        df['Ticket_count'] = ''
        df['Ticket_count'] = df['Ticket']

        print("✅ \n", df)
        df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Mlle', 'Ms','Countess','Dona'], 'Miss')
        df['Title'] = df['Title'].replace(['Mme','Lady'], 'Mrs')
        df.loc[~df['Title'].isin(['Miss', 'Mrs', 'Mr','Master']), 'Title'] = 'Other'

        df['Is_Married'] = 0
        df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
        df['Title'] = df['Title'].replace(['Mrs', 'Miss','Ms'], 'Mrs')
        
        encoder = OneHotEncoder(sparse=False)

        encoder.fit(df[self.cat_cols])
        encoded_df = encoder.transform(df[self.cat_cols])
        test_encoded_JR_df = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out(self.cat_cols))

        dfs = df.drop(self.cat_cols, axis=1)
        dfs = pd.concat([dfs, test_encoded_JR_df], axis=1)


        Jack_Rose_test_df = dfs.drop(['Name','Parch', 'Ticket', 'SibSp',
            'Cabin', 'Family_Size'], axis=1)
        for i in ['Ticket_count', 'Age', 'Fare']:
            if i == 'Ticket_count':
                Jack_Rose_test_df[i] = (Jack_Rose_test_df[i] - self.ticket_count_mean) / self.ticket_count_std
            elif i == 'Age':
                Jack_Rose_test_df[i] = (Jack_Rose_test_df[i] - self.age_mean) / self.age_std
            elif i == 'Fare':
                Jack_Rose_test_df[i] = (Jack_Rose_test_df[i] - self.fare_mean) / self.fare_std

        # 데이터프레임에 없는 칼럼을 0으로 설정
        for column in self.columns_to_add:
            if column not in Jack_Rose_test_df.columns:
                Jack_Rose_test_df[column] = 0

        result_df = Jack_Rose_test_df[self.columns_to_add]

        result_df.to_csv('./data/titanic/test_dfs.csv', index=False)

        print("⭐️ 전처리 후 데이터셋입니다.", result_df)

        return result_df
    