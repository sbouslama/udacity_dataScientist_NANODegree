import warnings
from scipy.spatial import KDTree

class Recommandation:
    def __init__(self, model,que_data, stu_data, pro_data,que_to_stu,pos_pairs,
                 que_proc: QueProc, pro_proc: ProProc):
        """
        data_path: path for all the datasets
        model: compiled model
        
        """
        self.que_feat_model =  Model(inputs=model.input[0], outputs=model.get_layer('lambda_160').output)
        self.pro_feat_model =  Model(inputs=model.input[1], outputs=model.get_layer('lambda_163').output)
        self.model = model
        # construct mappings from entity id to features
        self.que_dict = {row.values[0]: row.values[2:] for i, row in que_data.iterrows()}
        self.stu_dict = {stu: group.values[-1, 2:] for stu, group in stu_data.groupby('students_id')}
        self.pro_dict = {pro: group.values[-1, 2:] for pro, group in pro_data.groupby('professionals_id')}
        
        que_feat, que_ids, pro_feat, pro_ids = [], [],[],[]
        for que in self.que_dict.keys():
            cur_stu = que_to_stu[que]
            if cur_stu in self.stu_dict:
                # actual question's features are both question and student's features
                que_feat.append(np.hstack([self.stu_dict[cur_stu], self.que_dict[que]]))
                que_ids.append(que)
        
        for pro in self.pro_dict.keys():
            pro_feat.append(self.pro_dict[pro])
            pro_ids.append(pro)        
        
        self.que_ids = np.array(que_ids)
        self.que_feat = np.array(que_feat)
        self.pro_ids = np.array(pro_ids)
        self.pro_feat = np.array(pro_feat)
        
        # compute feature and question features:
#         self.pro_feat = self.pro_feat_model.predict([pro_feat])
#         self.que_feat = self.que_feat_model.predict([que_feat])
        
        self.pos_pairs = pos_pairs
        # initialize preprocessors
        self.que_proc = que_proc
        self.pro_proc = pro_proc
        
        # create KDTree trees from question and professional features
        self.que_tree = KDTree(self.que_feat)
        self.pro_tree = KDTree(self.pro_feat)
    
    
    def __get_que_feat(self, que_df, que_tags):
        """
            this function takes a question from a particular student and
            extracts it's features
        """
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])

        # extract and preprocess question's features
        que_feat = self.que_proc.transform(que_df, que_tags).values[:, 2:]
        
        # actual question's features are both question and student's features
        stu_feat = np.vstack([self.stu_dict[stu] for stu in que_df['questions_author_id']])
        que_feat = np.hstack([stu_feat, que_feat])      
        
#         que_feat = self.que_feat_model.predict(que_feat)
        return que_feat
    
    def __get_pro_feat(self, que_df, pro_df, ans_df, pro_tags):
        
        
        pro_df['professionals_date_joined'] = pd.to_datetime(pro_df['professionals_date_joined'])
        que_df['questions_date_added'] = pd.to_datetime(que_df['questions_date_added'])
        ans_df['answers_date_added'] = pd.to_datetime(ans_df['answers_date_added'])

        # extract and preprocess professional's features
        pro_feat = self.pro_proc.transform(pro_df, que_df, ans_df, pro_tags)

        # select the last available version of professional's features
        pro_feat = pro_feat.groupby('professionals_id').last().values[:, 1:]
#         pro_feat = self.pro_feat_model.predict(pro_feat)
        return pro_feat
    
    def __dict_to_df(self, que_dict):
        assert set(que_dict.keys()) - set([
            'questions_id','questions_author_id', 'questions_date_added','questions_title','questions_body','questions_tags'])==set([])
        df = pd.DataFrame.from_dict(que_dict)
        
        que_df = df[['questions_id','questions_author_id','questions_date_added','questions_title','questions_body']]
        tags_df = df[['questions_id','questions_tags']].rename(
            columns={'questions_id':'tag_questions_question_id', 'questions_tags':'tags_tag_name'}
        )
        
        tags_df['tags_tag_name'] = tags_df['tags_tag_name'].apply(lambda x: tp.process(x, allow_stopwords=True))
        que_df['questions_title'] = que_df['questions_title'].apply(tp.process)
        que_df['questions_body'] = que_df['questions_body'].apply(tp.process)
        que_df['questions_whole'] = que_df['questions_title'] + ' ' + que_df['questions_body']
        return que_df, tags_df
    
    def __convert_tuples(self, ids, tags):
        tuples = []
        for i, tgs in enumerate(tags):
            que = ids[i]
            for tag in tgs.split(' '):
                tuples.append((que, tag))
        return tuples
    
    def __convert_pro_dict(self, pro_dict):
        """
        Converts dictionary of professional data into desired form
        :param pro_dict: dictionary of professional data
        """
        # get DataFrame from dict
        pro_df = pd.DataFrame.from_dict(pro_dict)
        pros = pro_df['professionals_id'].values

        # create professional-tag tuples
        tuples = self.__convert_tuples(pro_df['professionals_id'].values,
                                            pro_df['professionals_subscribed_tags'].values)

        # create DataFrame from tuples
        pro_tags = pd.DataFrame(tuples, columns=['tag_users_user_id', 'tags_tag_name'])
        pro_df.drop(columns='professionals_subscribed_tags', inplace=True)

        pro_tags['tags_tag_name'] = pro_tags['tags_tag_name'].apply(lambda x: tp.process(x, allow_stopwords=True))
        pro_df['professionals_headline'] = pro_df['professionals_headline'].apply(tp.process)
        pro_df['professionals_industry'] = pro_df['professionals_industry'].apply(tp.process)

        return pro_df, pro_tags        
    
    def find_similar_questions(self, que_dict, top=10):
        que_df, tags_df = self.__dict_to_df(que_dict)
        #extract question features
        que_feat = self.__get_que_feat(que_df, tags_df)
        # calcule similarities  
        dists, ques = self.que_tree.query(que_feat, k=top)
        ques_ids = self.que_ids[ques]
        similar_questions = pd.DataFrame(list(zip(ques_ids[0], dists[0])), columns=['matched_id','distance'])
        return similar_questions
    
    def find_prof_by_question(self,que_dict, top=10):
        que_df, tags_df = self.__dict_to_df(que_dict)
        #extract question features
        que_feat = self.__get_que_feat(que_df, tags_df)
       
          # calcule similarities  
#         dists, pro = self.pro_tree.query(que_feat, k=top)
#         pro_ids = self.pro_ids[pro]
#         top_pro = pd.DataFrame(list(zip(pro_ids[0], dists[0])), columns=['matched_pro_id','distance'])


        prob = []
        for ele in self.pro_feat:
            prob.append(self.model.predict([que_feat.reshape(1,-1), ele.reshape(1,-1)])[0][0])
        prob = np.array(prob)
        indices = self.pro_ids[np.argsort(prob)[-top:]]
        top_pro = pd.DataFrame(list(zip(indices, prob[np.argsort(prob)[-top:]])), columns=['matched_pro_id','probability'])
        return top_pro
    
    
    def find_similar_professionals(self, prof_dict,que_df, ans_df, top=10):
        pro_df, pro_tags = self.__convert_pro_dict(prof_dict)
        #extract question features
        pro_feat = self.__get_pro_feat(que_df, pro_df, ans_df, pro_tags)
        # calcule similarities  
        dists, pros = self.pro_tree.query(pro_feat, k=top+1)
        pro_ids = self.pro_ids[pros]
        similar_pros = pd.DataFrame(list(zip(pro_ids[0][1:], dists[0][1:])), columns=['matched_pro_id','distance'])
        return similar_pros