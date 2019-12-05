import re
import sqlite3
import pandas as pd


class MessageWrangler():
    def __init__(self,
                 path_to_all_messages,
                 path_to_all_categories,
                 path_to_db,
                 categories_column,
                 delim):
        self.messages_df = pd.read_csv(path_to_all_messages)
        if not path_to_all_categories:
            path_to_all_categories = path_to_all_messages
        self.categories = pd.read_csv(path_to_all_categories)
        if not path_to_db:
            path_to_db = ':memory:'

        categories = self._categories_to_dummy(self.categories,
                                               categories_column,
                                               delim)

        full_data = self.messages_df.merge(categories, on='id')

        full_data.drop_duplicates(inplace=True)

        conn = xsqlite3.connect(path_to_db)

        full_data.to_sql('messages_info', conn)

    def _get_last_digit(self, string):
        return string[-1]

    def _categories_to_dummy(self, df, column, delim):
        new_df = df.copy()
        new_df_expanded = new_df[column].str.split(delim, expand=True)
        new_df = new_df[['id']].merge(new_df_expanded,
                                      left_index=True,
                                      right_index=True
                                      )
        column_names = new_df.iloc[0, 1:]

        pattern = re.compile('-\d')
        for i, name in enumerate(column_names):
            column_names[i] = re.sub(pattern, '', name)
        new_df.columns = ['id'] + column_names.tolist()

        new_df.iloc[:, 1:] = new_df.iloc[:, 1:].applymap(self._get_last_digit)

        return new_df


def load_messages_to_db(path_to_all_messages,
                        path_to_all_categories,
                        path_to_db,
                        categories_column='categories',
                        delim=';'):
    MessageWrangler(path_to_all_messages,
                    path_to_all_categories,
                    path_to_db,
                    categories_column,
                    delim
                    )
