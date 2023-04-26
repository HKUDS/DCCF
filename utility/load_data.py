import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp

class Data(object):
    def __init__(self, args):

        self.path = args.data_path + args.dataset
        self.n_batch = args.n_batch
        self.batch_size = args.batch_size
        self.train_num = args.train_num
        self.sample_num = args.sample_num

        train_file = self.path + '/train.txt'
        test_file = self.path + '/test.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')[1:]]

                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) > 0:
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, train_items = items[0], items[1:]

                        for i in train_items:
                            self.R[uid, i] = 1.
                        self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) > 0:
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        uid, test_items = items[0], items[1:]

                        self.test_set[uid] = test_items

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()
        return adj_mat

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()

    def uniform_sample(self):
        users = np.random.randint(0, self.n_users, int(self.n_batch * self.batch_size))
        train_data = []
        for i, user in tqdm(enumerate(users), desc='Sampling Data', total=len(users)):
            pos_for_user = self.train_items[user]
            pos_index = np.random.randint(0, len(pos_for_user))
            pos_item = pos_for_user[pos_index]
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if neg_item in pos_for_user:
                    continue
                else:
                    break
            train_data.append([user, pos_item, neg_item])
        self.train_data = np.array(train_data)
        return len(self.train_data)

    def mini_batch(self, batch_idx):
        st = batch_idx * self.batch_size
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data))
        batch_data = self.train_data[st: ed]
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items))
        return sta
