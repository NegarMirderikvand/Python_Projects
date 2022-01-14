import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
import csv, re, pickle


data = pd.read_csv("E:/Training/DSA/FinalProject/Persian-Sentiment-Analyzer-master/Persian-Sentiment-Analyzer-master/Tutorial_Dataset.csv")
reviews = data['Text']
rate = data['Score']
labels = data['Suggestion']


labels = np.array([1 if int(each)==1 or int(each)==2 else 0 for each in labels])
#cleaning dataset
words=[]
all_text = ''

for t in range (len(reviews)):
	text = reviews[t]
	text = text.replace('\u200c',' ')
	text = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', text)
	all_text += text
	all_text += ' '
	wordsInText = text.split()
	for word in wordsInText:
		if word != ' ' or word != '':
			words.append(word)


counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

with open("mySavedDict.txt", "wb") as myFile:
    pickle.dump(vocab_to_int, myFile)

'''
with open("mySavedDict.txt", "rb") as myFile:
    myNewPulledInDictionary = pickle.load(myFile)
'''
#print myNewPulledInDictionary


reviews_ints = []
for each in reviews:
	#print (each)
	each = each.replace('\u200c',' ')
	each = re.sub(r'[^a-zA-Z0-9آ-ی۰-۹ ]', ' ', each)
	reviews_ints.append([vocab_to_int[word] for word in each.split()])


'''
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
'''

seq_len = 400
features = np.zeros((len(reviews), seq_len), dtype=int)
for i, row in enumerate(reviews_ints):
	print (i , row)
	print ('****')
	features[i, -len(row):] = np.array(row)[:seq_len]

split_frac = 0.9
split_idx = int(len(features)*split_frac)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

lstm_size = 256
lstm_layers = 1
batch_size = 200
learning_rate = 0.001

n_words = len(vocab)
print (n_words)

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

embed_size = 500 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words+1, embed_size), -1, 1))
    #print ('len input   :   ', inputs_)
    embed = tf.nn.embedding_lookup(embedding, inputs_)
    #print (embed[46421])

with graph.as_default():
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
	initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
	outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


with graph.as_default():
	predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
	cost = tf.losses.mean_squared_error(labels_, predictions)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with graph.as_default():
	correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batches(x, y, batch_size=100):
	n_batches = len(x)//batch_size
	x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
	for ii in range(0, len(x), batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]

epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
	sess.run(tf.global_variables_initializer())
	iteration = 1
	for e in range(epochs):
		state = sess.run(initial_state)
		for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
			print (embed[46421])
			feed = {inputs_: x, labels_: y[:, None], keep_prob: 0.5, initial_state: state}
			loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
			if iteration%5==0:
				print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))
			if iteration%25==0:
				val_acc = []
				val_state = sess.run(cell.zero_state(batch_size, tf.float32))
				for x, y in get_batches(val_x, val_y, batch_size):
					feed = {inputs_: x,labels_: y[:, None],keep_prob: 1,initial_state: val_state}
					batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
					val_acc.append(batch_acc)
				print("Val acc: {:.3f}".format(np.mean(val_acc)))
			iteration +=1
	saver.save(sess, "checkpoints/sentiment.ckpt")

#46421

