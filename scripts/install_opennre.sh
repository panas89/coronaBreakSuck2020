# start with home directory
cd

# basic install
git clone https://github.com/thunlp/OpenNRE.git
cd OpenNRE
echo 'Installing openre package...'
python setup.py install 

# Download the pretrain model
wget -P bert-base-uncased https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/bert-base-uncased/config.json
wget -P bert-base-uncased https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/bert-base-uncased/pytorch_model.bin
wget -P bert-base-uncased https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/pretrain/bert-base-uncased/vocab.txt

# create neccessary folder for the pretrain model so that we can run the model smoothly
mkdir -p ~/.opennre2/pretrain/bert-base-uncased/

# move the pretrain model to the corresponding location for the library, i.e., being with HOME/
mv -rf bert-base-uncased/* ~/.opennre2/pretrain/bert-base-uncased/

# return to home; done
cd












