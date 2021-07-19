from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from nltk.tokenize import sent_tokenize
import numpy as np

paragraph = """The current Coronavirus pandemic strengthens a type of narrow-minded nationalism that
the international community seemed to have already left behind. During the last decades
since the end of the Cold War, some considerable progress has been made to establish
transnational institutions which are capable of tackling the great challenges of our time.
Now, however, national instincts return precisely at a moment when the spirit of
transnational cooperation is needed more than in less demanding times. At the first peak of
the crisis in March 2020, even some of the well-established international rules and
agreements were on the verge of collapsing. What would be appropriate, instead, is a
considerable increase of global cooperation. Furthermore, in my view, the step to be taken
in this situation is to establish transnational institutions which cannot simply be ruled out
when resentments of nationalism occasionally re-emerge. I think of firm and stable
institutions of global crisis management.
The position I want to argue against is normally not explicitly defended. It is rather an
implicit one but one that is largely shared: I call it the dogma of nationalism in politics. It is
hardly disputed by anyone that the nation-state and the national community is the ultimate
foundation to organize politics. But this dogma leads us into a highly ineffective and
unwelcome global situation when it comes to transnational problems like the current
pandemic.
One phenomenon that makes this palpable is the shutdown of national borders at the
beginning of the Coronavirus crisis. Even in EU Europe, including the Schengen Area, national
border controls have been widely re-established. The new inner-European border controls
were extremely strict and almost insurmountable, at least for the majority of citizens,
including unmarried couples living on both sides of a border. The spirit of free travel and free
trade broke down within only a few days. And quite surprisingly, almost nobody protested
against it. Strictly speaking, however, the line of action taken by national administrations
without Brussels being involved violates EU law and neglects the sense of the contracts: the
border closure was not multilaterally agreed upon but a simple unilateral decision of each
single nation-state. The respective neighbouring countries were not asked for their consent,
sometimes they were not even informed before the measure was taken. What is worse:
concerning the fight against Covid-19, the shutdown of inner EU borders had no positive
effect at all – it did not even improve the chances to control the spread of the disease. In
order to limit the dissemination of the virus, one has to trace single cases under a local and
regional supervision. A lockdown that reduces social contacts does not necessarily include
the control of borders. It isn’t helpful to close the border between Germany and Poland to
successfully reduce the number of Covid-19 cases in Görlitz and Zgorzelec. Rather, this was a
symbolic measure, symbolic for a robust defence of the national interests, undertaken by
the national governments to calm the fear and resentment of the broader public. It is an
expression of the dogma of nationalism. """

sentences = sent_tokenize(paragraph) # spliting paragraph into individual sentences
input_data = []
output_data = []
max_len = 40  # this is the max length of any sub sequence used to predict the next character
for sentence1 in sentences:
    for i in range(0, len(sentence1)-max_len):
        input_data.append(sentence1[i:i+max_len])
        output_data.append(sentence1[i+max_len])
print(input_data[0])
print(output_data[0])

vocablury = sorted(set(paragraph))
char_to_id={char:idx for idx,char in enumerate(vocablury)}
id_to_char={idx:char for idx,char in enumerate(vocablury)}
x=np.zeros((len(input_data),max_len,len(vocablury)),dtype='float32')
y=np.zeros((len(output_data),len(vocablury)),dtype='float32')

for s_idx,sequence in enumerate(input_data):
    for idx,char in enumerate(sequence):
        x[s_idx,idx,char_to_id[char]]=1
    y[s_idx,char_to_id[output_data[s_idx]]]=1
model = Sequential()
model.add(LSTM(128,input_shape=(max_len,len(vocablury))))
model.add(Dense(len(vocablury),activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()
model.fit(x,y, batch_size=128,epochs=15,validation_split=0.2)


""" to be predicted """
tobecompleted = "ThcurrntCndmic"
X_test=np.zeros((1,max_len,len(vocablury)),dtype='float32')
for t,char in enumerate(tobecompleted):
    X_test[0,t,char_to_id[char]]=1
pred=model.predict(X_test,verbose=0)
prob_next_char=pred[0]
next_index=np.argmax(prob_next_char)
next_char= id_to_char[next_index]
print(next_char+"ds")
print(output_data.count("e"))
print(len(output_data))
