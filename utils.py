import ast


class Utils:
    @staticmethod
    def clean_file(file):
        clean_list = []
        if file.endswith('.jsonl.tmp'):
            with open(file, encoding='utf-8', mode='r') as f:
                for line in f:
                    data = ast.literal_eval(line)
                    if 'gender' in data.keys() and data['gender']:
                        for review in data['reviews']:
                            if review['rating'] and review['text']:
                                new_data = {
                                    'text': review['text'][0],
                                    'rating': int(review['rating']),
                                    'gender': data['gender']
                                }
                                clean_list.append(new_data)
        return clean_list
