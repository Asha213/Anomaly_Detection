import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random

# model imports
from sklearn.ensemble import RandomForestClassifier
# processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
def process(trainpath,testpath):
    df = pd.read_csv( trainpath, header=None )
    test_df = pd.read_csv( testpath, header=None )
    columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'])

    df.columns = columns
    test_df.columns = columns
    # map normal to 0, all attacks to 1
    is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
    test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

    #data_with_attack = df.join(is_attack, rsuffix='_flag')
    df['attack_flag'] = is_attack
    test_df['attack_flag'] = test_attack
    # lists to hold our attack classifications
    dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
    probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
    privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
    access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

    # we will use these for plotting below
    attack_labels = ['Normal','DoS','Probe','Privilege','Access']
    def map_attack(attack):
        if attack in dos_attacks:
            # dos_attacks map to 1
            attack_type = 1
        elif attack in probe_attacks:
            # probe_attacks mapt to 2
            attack_type = 2
        elif attack in privilege_attacks:
            # privilege escalation attacks map to 3
            attack_type = 3
        elif attack in access_attacks:
            # remote access attacks map to 4
            attack_type = 4
        else:
            # normal maps to 0
            attack_type = 0
            
        return attack_type

    # map the data and join to the data set
    attack_map = df.attack.apply(map_attack)
    df['attack_map'] = attack_map

    test_attack_map = test_df.attack.apply(map_attack)
    test_df['attack_map'] = test_attack_map

    # get the intial set of encoded features and encode them
    features_to_encode = ['protocol_type', 'service', 'flag']
    encoded = pd.get_dummies(df[features_to_encode])
    test_encoded_base = pd.get_dummies(test_df[features_to_encode])

    # not all of the features are in the test set, so we need to account for diffs
    test_index = np.arange(len(test_df.index))
    column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))

    diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

    # we'll also need to reorder the columns to match, so let's get those
    column_order = encoded.columns.to_list()

    # append the new columns
    test_encoded_temp = test_encoded_base.join(diff_df)

    # reorder the columns
    test_final = test_encoded_temp[column_order].fillna(0)

    # get numeric features, we won't worry about encoding these at this point
    numeric_features = ['duration', 'src_bytes', 'dst_bytes']

    # model to fit/test
    to_fit = encoded.join(df[numeric_features])
    test_set = test_final.join(test_df[numeric_features])
    # create our target classifications
    binary_y = df['attack_flag']
    multi_y = df['attack_map']

    test_binary_y = test_df['attack_flag']
    test_multi_y = test_df['attack_map']

    # build the training sets
    binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, binary_y, test_size=0.6)
    multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, multi_y, test_size = 0.6)
    # model for the binary classification
    binary_model = RandomForestClassifier()
    binary_model.fit(binary_train_X, binary_train_y)
    binary_predictions = binary_model.predict(binary_val_X)

    # calculate and display our base accuracty
    base_rf_score = accuracy_score(binary_predictions,binary_val_y)
    print("Random Forest Binary Classification Accuracy",base_rf_score)
    # model for the mulit classification
    multi_model = RandomForestClassifier()
    multi_model.fit(multi_train_X, multi_train_y)
    multi_predictions = multi_model.predict(multi_val_X)

    # get the score
    acc=accuracy_score(multi_predictions,multi_val_y)
    print("Random Forest Multimodel Prediction=",acc)

#process("./corrected","./kddcup.data_10_percent_corrected")

