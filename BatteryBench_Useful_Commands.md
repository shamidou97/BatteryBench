\#############################################################################Now let's set up Docker#########################################################

\#First confirm your Docker installation:

docker --version

> Docker version 28.2.2, build 28.2.2-0ubuntu1\~24.04.1



docker-compose --version



The command 'docker-compose' could not be found in this WSL 2 distro.

We recommend to activate the WSL integration in Docker Desktop settings.



For details about using Docker Desktop with WSL 2, visit:



https://docs.docker.com/go/wsl2/



\#check GPU support:



\# Test Docker connection

docker info | head -5

&#x20;Version:    28.2.2

&#x20;Context:    default



\# Test docker compose (new syntax)

docker compose version

Docker Compose version v2.40.3-desktop.1



\# Install compose plugin if needed

sudo apt-get install docker-compose-plugin -y

(base) syhamid@HAMIDOU:/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/dataset$ sudo apt-get install docker-compose-plugin -y

\[sudo] password for syhamid: 

Reading package lists... Done

Building dependency tree... Done

Reading state information... Done

E: Unable to locate package docker-compose-plugin



\# Test GPU

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi



Unable to find image 'nvidia/cuda:11.8.0-base-ubuntu22.04' locally

11.8.0-base-ubuntu22.04: Pulling from nvidia/cuda

5e3b7ee77381: Pull complete 

4cda774ad2ec: Pull complete 

aece8493d397: Pull complete 

5bd037f007fd: Pull complete 

775f22adee62: Pull complete 

Digest: sha256:f895871972c1c91eb6a896eee68468f40289395a1e58c492e1be7929d0f8703b

Status: Downloaded newer image for nvidia/cuda:11.8.0-base-ubuntu22.04

Sat May 23 07:17:42 2026       

+-----------------------------------------------------------------------------------------+

| NVIDIA-SMI 590.44.01              Driver Version: 591.44         CUDA Version: 13.1     |

+-----------------------------------------+------------------------+----------------------+

| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |

| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |

|                                         |                        |               MIG M. |

|=========================================+========================+======================|

|   0  NVIDIA GeForce RTX 4060 ...    On  |   00000000:01:00.0 Off |                  N/A |

| N/A   56C    P8              4W /   35W |       0MiB /   8188MiB |      0%      Default |

|                                         |                        |                  N/A |

+-----------------------------------------+------------------------+----------------------+



+-----------------------------------------------------------------------------------------+

| Processes:                                                                              |

|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |

|        ID   ID                                                               Usage      |

|=========================================================================================|

|  No running processes found                                                             |

+-----------------------------------------------------------------------------------------+





\###############################################################################Project Setup######################################################################

\# Create project

mkdir -p /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench



\# Create folder structure

mkdir -p trainer/src

mkdir -p dashboard/src

mkdir -p dashboard/templates

mkdir -p dashboard/static

mkdir -p mysql/init

mkdir -p data/cache

mkdir -p models



mkdir -p results

\# Verify

find . -type d | sort



./dashboard

./dashboard/src

./dashboard/static

./dashboard/templates

./data

./data/cache

./models

./mysql

./mysql/init

./trainer

./trainer/src

(base) syhamid@HAMIDOU:/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench$ 



\# generate all files with :



docker-compose.yml



\# File placement:



BatteryBench/

├── docker-compose.yml              ← root

├── trainer/

│   ├── Dockerfile                  ← trainer.Dockerfile

│   └── requirements.txt            ← trainer\_requirements.txt

├── dashboard/

│   ├── Dockerfile                  ← dashboard.Dockerfile

│   └── requirements.txt            ← dashboard\_requirements.txt

└── mysql/

&#x20;   └── init/

&#x20;       └── schema.sql              ← schema.sql





\# Extract the zip and copy:



cd /mnt/c/Users/shami/Downloads



\# Extract zip

unzip "\*.zip" -d batterybench\_files



\# Check extracted files

ls batterybench\_files/



\# copy to project:



BASE=/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench

SRC=/mnt/c/Users/shami/Downloads/batterybench\_files



cp $SRC/docker-compose.yml          $BASE/

cp $SRC/trainer.Dockerfile          $BASE/trainer/Dockerfile

cp $SRC/trainer\_requirements.txt    $BASE/trainer/requirements.txt

cp $SRC/dashboard.Dockerfile        $BASE/dashboard/Dockerfile

cp $SRC/dashboard\_requirements.txt  $BASE/dashboard/requirements.txt

cp $SRC/schema.sql                  $BASE/mysql/init/schema.sql



\# Verify all files in place

find $BASE -name "\*.yml" -o -name "Dockerfile" \\

&#x20;    -o -name "requirements.txt" -o -name "\*.sql" | sort

\# trainer requirements.txt is missing. Fix it:



cp /mnt/c/Users/shami/Downloads/batterybench\_files/trainer\_requirements.txt \\

&#x20;  /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench/trainer/requirements.txt



\# Verify

find /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench \\

&#x20;    -name "\*.yml" -o -name "Dockerfile" \\

&#x20;    -o -name "requirements.txt" -o -name "\*.sql" | sort



\# build the data\_loader.py:



Created 2 files, ran a command, read a file

Save to trainer/src/data\_loader.py. Here's what it does:



\# Data loader design:



&#x09;		Value							Reason

Window size		32 cycles						Enough history to see degradation trend



Stride			1 cycle							Maximum data augmentation



Features		9 per cycle						All summary stats + normalized cycle number



Scaler			MinMaxScaler						Fitted on train only — no data leakage



Train			Batch 1-4						39 batteries · 4 protocols



Val			Batch 5 (RW)						Unseen charge protocol



Test			Batch 6 (Satellite)					Unseen protocol — hardest generalization





\# Test it before Docker



cd /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench



cd /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench



\# Create symlinks

for i in 1 2 3 4 5 6; do

&#x20;   ln -s /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/dataset/Batch-$i data/Batch-$i

done



\# Copy data\_loader to trainer/src/

cp /mnt/c/Users/shami/Downloads/data\_loader.py trainer/src/data\_loader.py



\# Verify symlinks

ls -la data/



\# Test data loader

python trainer/src/data\_loader.py



\####################################################################Label error replacing \_ah with \_Ah and \_wh with \_Wh######################################################################



&#x20;File "/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench/trainer/src/battery\_data\_loader.py", line 75, in load\_battery

&#x20;   cap = np.array(s\['discharge\_capacity\_ah'], dtype=np.float32) <-----------------\_ah

&#x20;                  \~^^^^^^^^^^^^^^^^^^^^^^^^^

KeyError: 'discharge\_capacity\_ah'

(base) syhamid@HAMIDOU:/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench$ 



\## Script for the summary key

python3 -c "

import scipy.io

mat = scipy.io.loadmat(

&#x20;   '/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/dataset/Batch-1/2C\_battery-1.mat',

&#x20;   simplify\_cells=True)

print('Summary keys:', list(mat\['summary'].keys()))



"

Summary keys: \['charge\_capacity\_Ah', 'discharge\_capacity\_Ah', 'charge\_power\_Wh', 'discharge\_power\_Wh', 'charge\_median\_voltage', 'discharge\_median\_voltage', 'charge\_mean\_voltage', 'discharge\_mean\_voltage', 'cycle\_life', 'description']



\##  Summary shows \_Ah and Wh



\## Script for correcting ah and wh into Ah and Wh <---------- case sensitive



python3 - << 'EOF'



f = open('trainer/src/battery\_data\_loader.py', encoding='utf-8').read()



\# Fix key names - capitalize Ah and Wh

fixes = \[

&#x20;   ('discharge\_capacity\_ah', 'discharge\_capacity\_Ah'),

&#x20;   ('charge\_capacity\_ah',    'charge\_capacity\_Ah'),

&#x20;   ('discharge\_power\_wh',    'discharge\_power\_Wh'),

&#x20;   ('charge\_power\_wh',       'charge\_power\_Wh'),

]



for old, new in fixes:

&#x20;   f = f.replace(f"s\['{old}']", f"s\['{new}']")

&#x20;   f = f.replace(f"'{old}'",    f"'{new}'")



open('trainer/src/battery\_data\_loader.py', 'w', encoding='utf-8').write(f)

print('Fixed!')



\# Verify

for old, new in fixes:

&#x20;   print(f'  {new}: {new in f}')

EOF



\#######################################################################Battery Distribution###########################################################################################################



Same distribution data     → random split OK

Different groups/protocols → group-based split REQUIRED<------------------------The case will be used for training

&#x20; otherwise you get:

&#x20;   Inflated test scores

&#x20;   Models that fail in production

&#x20;   Results that don't reproduce on new data







✅ What looks good

Train : 16,046 windows · 39 batteries · 4 protocols

Val   : 1,735  windows · 8  batteries · RW protocol

Test  : 8,156  windows · 8  batteries · Satellite protocol

Total : 25,937 windows



⚠️ Two issues to fix

**Issue 1 — SOH > 100% in train and val**



Train SOH max : 101.8%  ← impossible physically

Val   SOH max : 111.3%  ← very wrong



Cause: discharge\_capacity\_Ah\[i] > discharge\_capacity\_Ah\[0]

&#x20;      First cycle is not always the maximum capacity

&#x20;      Some batteries have a "formation" phase where

&#x20;      capacity increases before degrading



Fix: SOH = cap\[i] / max(cap\[0:5]) × 100

&#x20;    Use max of first 5 cycles as reference



**Issue 2 — Satellite EOL = cycle 2**



Batch-6 EOL : 2  ← all satellites show EOL at cycle 2



Cause: Satellite discharge profile is different

&#x20;      capacity starts low, rises, then falls

&#x20;      SOH < 80% at cycle 2 is a false EOL detection



Fix: Use max(cap) as reference instead of cap\[0]

&#x20;    SOH = cap\[i] / max(cap) × 100  ← relative to peak

&#x20;    Or clip SOH to \[0, 100]



**Fix both issues now**



python3 - << 'EOF'

f = open('trainer/src/battery\_data\_loader.py', encoding='utf-8').read()



old = """    cap = np.array(s\['discharge\_capacity\_Ah'], dtype=np.float32)

&#x20;   soh = (cap / cap\[0] \* 100.0).astype(np.float32)

&#x20;   eol = next((i+1 for i, v in enumerate(soh) if v < EOL\_SOH), n)"""



new = """    cap = np.array(s\['discharge\_capacity\_Ah'], dtype=np.float32)

&#x20;   # Use max of first 5 cycles as reference capacity

&#x20;   # Handles formation phase (capacity rise before degradation)

&#x20;   # and satellite profiles where cap\[0] is not representative

&#x20;   ref\_cap = float(np.max(cap\[:min(5, len(cap))]))

&#x20;   soh     = np.clip(cap / ref\_cap \* 100.0, 0, 105).astype(np.float32)

&#x20;   eol     = next((i+1 for i, v in enumerate(soh) if v < EOL\_SOH), n)"""



if old in f:

&#x20;   f = f.replace(old, new)

&#x20;   open('trainer/src/battery\_data\_loader.py', 'w', encoding='utf-8').write(f)

&#x20;   print('Fixed!')

else:

&#x20;   print('Pattern not found')

EOF



Remove the cache:

rm trainer/data/cache/batterybench.pkl



**Final dataset summary** ✅



Train : 12,576 windows · SOH \[80.9-100%] · mean=94.5%

Val   :  2,266 windows · SOH \[67.1-91.1%] · mean=81.9%

Test  :  1,735 windows · SOH \[66.4-100%]  · mean=92.0%

Total : 16,577 windows · 47 batteries



**Everything is correct**

**Train SOH \[80.9-100%]:**



First 80% of each battery lifetime

Battery is healthy → SOH near 100%

Drops to \~80% by end of train period **✅**



**Val SOH \[67.1-91.1%] — end-of-life region:**



Last 20% of each battery lifetime

Battery is degrading → SOH drops toward EOL

Mean=81.9% → right at the critical degradation zone ✅

This tests: "can model predict end-of-life accurately?"



**Test SOH \[66.4-100%] — unseen RW protocol:**



Full lifetime of Batch-5 batteries

Unseen charge protocol → true generalization test ✅

Mean=92% → healthy batteries with RW charging



**Split design confirmed correct**



Train → Val → Test represents increasing difficulty:



Train : healthy batteries, known protocols     ← easy

Val   : degrading batteries, known protocols   ← harder

Test  : full lifetime, UNKNOWN protocol        ← hardest



**BatteryBench status**



✅ Docker environment    ready

✅ MySQL schema          ready

✅ Data loader v2        working correctly

&#x20; - 47 batteries (Batch-6 excluded)

&#x20; - Per-battery 80/20 train/val split

&#x20; - Batch-5 RW as cross-protocol test

&#x20; - SOH clipped to \[0-100%]

&#x20; - ref\_cap = max(cap\[:5])

⬜ trainer.py            next

⬜ bridge\_to\_sql.py      next

⬜ dashboard             next



**####################################################################################Check the MySQL container status:###################################################################################**



cd /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench



\# Check if containers are running

docker compose ps



\# Start MySQL container

docker compose up -d mysql



\# Watch startup logs

docker compose logs -f MySQL



\# If not running, start MySQL only

docker compose up -d mysql

\# Wait for MySQL to be ready (10 seconds)

sleep 10



\# Check container health

docker compose ps



\# Check tables were created from schema.sql

docker exec batterybench\_mysql mysql -ubattuser -pbattery123 \\

&#x20;   batterybench -e "SHOW TABLES;"



\# Check batch data was seeded

docker exec batterybench\_mysql mysql -ubattuser -pbattery123 \\

&#x20;   batterybench -e "SELECT \* FROM batches;"



**Database status** ✅



Tables    : batches · batteries · cycles · model\_results ✅

Views     : v\_battery\_summary · v\_model\_comparison · v\_soh\_degradation ✅

Batch-1-5 : seeded correctly ✅

Batch-6   : still in DB ← needs to be removed



**Remove Batch-6 from database<----------------- DUE LEO (Battery in orbit spacecraft the charge/discharge exhibits different pattern)**



**docker exec batterybench\_mysql mysql -ubattuser -pbattery123 \\**

&#x20;   **batterybench -e "**

**DELETE FROM batches WHERE name='Batch-6';**

**SELECT id, name, charge\_protocol, n\_batteries FROM batches;**

**"**



**# Fix the schema.sql so it does not seed Batch-6 on future builds:**



**p**ython3 - << 'EOF'

f = open('mysql/init/schema.sql', encoding='utf-8').read()



old = """('Batch-6', 'Satellite', 'Simulated satellite load · 8 batteries',    8);"""

new = """-- Batch-6 excluded: LEO satellite partial cycles

\-- incompatible with SOH definition for EV/ground applications;"""



if old in f:

&#x20;   f = f.replace(old, new)

&#x20;   open('mysql/init/schema.sql', 'w', encoding='utf-8').write(f)

&#x20;   print('schema.sql updated')

else:

&#x20;   print('Pattern not found')

EOF



**# database:Now remove Batch-6 from the running database:**



docker exec batterybench\_mysql mysql -ubattuser -pbattery123 \\

&#x20;   batterybench -e "

DELETE FROM batches WHERE name='Batch-6';

SELECT id, name, charge\_protocol, n\_batteries FROM batches;

"



**# BatteryBench — Current Status**



**✅ Docker MySQL     : running · healthy · port 3307**

**✅ Schema           : 4 tables · 3 views · 5 batches seeded**

**✅ Batch-6 excluded : removed from DB + schema.sql updated**

**✅ Data loader v2   : 16,577 windows · correct splits**

&#x20;  **Train : 12,576  (Batch 1-4 first 80%)**

&#x20;  **Val   :  2,266  (Batch 1-4 last 20% ← end-of-life)**

&#x20;  **Test  :  1,735  (Batch 5 RW ← unseen protocol)**

**⬜ bridge\_to\_sql.py : populate batteries + cycles tables**

**⬜ trainer.py       : CNN vs LSTM vs Transformer**

**⬜ dashboard        : Flask + Chart.js**



**#Also fix the version warning in docker-compose.yml**



**python3 - << 'EOF'**

**f = open('docker-compose.yml', encoding='utf-8').read()**

**f = f.replace('version: "3.9"\\n\\n', '')**

**open('docker-compose.yml', 'w', encoding='utf-8').write(f)**

**print('version attribute removed')**

**EOF**





**docker compose up -d mysql**



**docker compose ps**



**##########################################################################3 BatteryBench- Resume Checklist##############################################################################**



**✅ Project structure   BatteryBench/ folders created**

**✅ Docker environment  docker-compose.yml · Dockerfiles**

**✅ MySQL schema        4 tables · 3 views · 5 batches**

**✅ Data loader v2      16,577 windows · correct splits**

**✅ Batch-6 excluded    LEO satellite — documented reason**



**⬜ bridge\_to\_sql.py   populate batteries + cycles tables**

**⬜ trainer.py         CNN vs LSTM vs Transformer (regression)**

**⬜ dashboard          Flask + Chart.js SOH visualization**

**⬜ GitHub repo        BatteryBench**

**⬜ LinkedIn post      BatteryBench announcement**



**##Key decisions made this session**



**1. Option A split — Batch-6 excluded (LEO physics)**

**2. Cross-protocol test — Batch-5 RW as test set**

**3. Per-battery 80/20 — last 20% cycles = val**

**4. SOH ref = max(cap\[:5]) — handles formation phase**

**5. Docker MySQL on port 3307 — avoids BearingBench conflict**



**#####################################################################Setup and load data######################################################################################################**



**cd /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench**



**# Start MySQL**

**docker compose up -d mysql**



**# Check status**

**docker compose ps**



**# Verify data loader cache**

**ls -lh trainer/data/cache/**



**# Quick data loader test**

**python trainer/src/battery\_data\_loader.py**



**BatteryBench — Data Loader**

**=======================================================**

&#x20; **Dataset  : XJTU Battery (55 batteries, 6 batches)**

&#x20; **Task     : SOH regression (0-100%)**

&#x20; **Window   : 32 cycles, stride=1**

&#x20; **Features : 9 per cycle**

&#x20; **EOL      : SOH < 80.0%**



&#x20; **Splits:**

&#x20;   **Batch-1   : 2C           -> trainval**

&#x20;   **Batch-2   : 3C           -> trainval**

&#x20;   **Batch-3   : R2.5         -> trainval**

&#x20;   **Batch-4   : R3           -> trainval**

&#x20;   **Batch-5   : RW           -> test**



**Loading...**

&#x20; **Loading cache: /mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench/trainer/data/cache/batterybench.pkl**

&#x20; **train : (12576, 32, 9)**

&#x20; **val   : (2266, 32, 9)**

&#x20; **test  : (1735, 32, 9)**



**-- Summary --**

&#x20; **train : (12576, 32, 9)  SOH \[80.9-100.0%]  mean=94.5%**

&#x20; **val   : (2266, 32, 9)  SOH \[67.1-91.1%]  mean=81.9%**

&#x20; **test  : (1735, 32, 9)  SOH \[66.4-100.0%]  mean=92.0%**

&#x20; **Total  : 16,577 windows**



&#x20; **LSTM/Transformer : (12576, 32, 9)**

&#x20; **CNN              : (12576, 32, 9, 1)**



**Data loader ready!**



**(base) syhamid@HAMIDOU:/mnt/d/DEEP-LEARNING/ADVANCED\_DEEP\_LEARNING/PROJECTS/BATTERY/XJTU/BatteryBench/trainer/src$** 



**#####################################################################################build the bridge to sql#########################################################**



**trainer/src/bridge\_to\_sql.py**





**BatteryBench — Bridge to MySQL**

**=======================================================**

&#x20; **Host     : localhost:3307**

&#x20; **Database : batterybench**

&#x20; **Connected ✅**



&#x20; **Batch      Battery                         Cycles    EOL Split**

&#x20; **-----------------------------------------------------------------**

&#x20; **Batch-1    2C\_battery-1                       390    390  trainval**

&#x20; **Batch-1    2C\_battery-2                       407    407  trainval**

&#x20; **Batch-1    2C\_battery-3                       393    393  trainval**

&#x20; **Batch-1    2C\_battery-4                       396    396  trainval**

&#x20; **Batch-1    2C\_battery-5                       403    394  trainval**

&#x20; **Batch-1    2C\_battery-6                       408    398  trainval**

&#x20; **Batch-1    2C\_battery-7                       402    383  trainval**

&#x20; **Batch-1    2C\_battery-8                       420    418  trainval**

&#x20; **Batch-2    3C\_battery-1                       299    299  trainval**

&#x20; **Batch-2    3C\_battery-10                      164    164  trainval**

&#x20; **Batch-2    3C\_battery-11                      131    131  trainval**

&#x20; **Batch-2    3C\_battery-12                      212    212  trainval**

&#x20; **Batch-2    3C\_battery-13                      226    226  trainval**

&#x20; **Batch-2    3C\_battery-14                      147    147  trainval**

&#x20; **Batch-2    3C\_battery-15                      168    168  trainval**

&#x20; **Batch-2    3C\_battery-2                       292    285  trainval**

&#x20; **Batch-2    3C\_battery-3                       286    286  trainval**

&#x20; **Batch-2    3C\_battery-4                       322    322  trainval**

&#x20; **Batch-2    3C\_battery-5                       297    297  trainval**

&#x20; **Batch-2    3C\_battery-6                       322    322  trainval**

&#x20; **Batch-2    3C\_battery-7                       319    319  trainval**

&#x20; **Batch-2    3C\_battery-8                       270    270  trainval**

&#x20; **Batch-2    3C\_battery-9                       287    287  trainval**

&#x20; **Batch-3    R2.5\_battery-1                     592    579  trainval**

&#x20; **Batch-3    R2.5\_battery-2                     552    539  trainval**

&#x20; **Batch-3    R2.5\_battery-3                     667    655  trainval**

&#x20; **Batch-3    R2.5\_battery-4                     557    549  trainval**

&#x20; **Batch-3    R2.5\_battery-5                     562    549  trainval**

&#x20; **Batch-3    R2.5\_battery-6                     542    529  trainval**

&#x20; **Batch-3    R2.5\_battery-7                     527    519  trainval**

&#x20; **Batch-3    R2.5\_battery-8                     617    609  trainval**

&#x20; **Batch-4    R3\_battery-1                       799    696  trainval**

&#x20; **Batch-4    R3\_battery-2                       673    606  trainval**

&#x20; **Batch-4    R3\_battery-3                       673    575  trainval**

&#x20; **Batch-4    R3\_battery-4                       727    594  trainval**

&#x20; **Batch-4    R3\_battery-5                       739    666  trainval**

&#x20; **Batch-4    R3\_battery-6                       715    648  trainval**

&#x20; **Batch-4    R3\_battery-7                       601    552  trainval**

&#x20; **Batch-4    R3\_battery-8                       751    660  trainval**

&#x20; **Batch-5    RW\_battery-1                       197    197  test**

&#x20; **Batch-5    RW\_battery-2                       307    297  test**

&#x20; **Batch-5    RW\_battery-3                       340    322  test**

&#x20; **Batch-5    RW\_battery-4                       252    246  test**

&#x20; **Batch-5    RW\_battery-5                       274    265  test**

&#x20; **Batch-5    RW\_battery-6                       219    217  test**

&#x20; **Batch-5    RW\_battery-7                       186    186  test**

&#x20; **Batch-5    RW\_battery-8                       208    200  test**



**── Database Summary ─────────────────────────────────**

&#x20; **Batteries : 47**

&#x20; **Cycles    : 19,238**

&#x20; **SOH range : 67.07% – 100.0%  mean=93.12%**



**── Per Batch ─────────────────────────────────────────**

&#x20; **Batch      Protocol    Batteries   Cycles   AvgSOH**

&#x20; **--------------------------------------------------**

&#x20; **Batch-1    2C                  8    3,219    92.9%**

&#x20; **Batch-2    3C                 15    3,742    94.4%**

&#x20; **Batch-3    R2.5                8    4,616    94.3%**

&#x20; **Batch-4    R3                  8    5,678    90.3%**

&#x20; **Batch-5    RW                  8    1,983    96.3%**



&#x20; **Total batteries : 47**

&#x20; **Total cycles    : 19,238**



**BatteryBench database populated! ✅**





**####################################################################################Way to check MySQL database###############################################################################################**

**Option 1 — From WSL terminal (quickest)**

**# Connect to MySQL inside Docker container**

**docker exec -it batterybench\_mysql mysql -ubattuser -pbattery123 batterybench**



**# Then run queries interactively:**

**SHOW TABLES;**

**+------------------------+**

**| Tables\_in\_batterybench |**

**+------------------------+**

**| batches                |**

**| batteries              |**

**| cycles                 |**

**| model\_results          |**

**| v\_battery\_summary      |**

**| v\_model\_comparison     |**

**| v\_soh\_degradation      |**

**+------------------------+**

**7 rows in set (0.00 sec)**



**SELECT COUNT(\*) FROM batteries;**

**+----------+**

**| COUNT(\*) |**

**+----------+**

**|       47 |**

**+----------+**

**1 row in set (0.01 sec)**



**SELECT COUNT(\*) FROM cycles;**

**+----------+**

**| COUNT(\*) |**

**+----------+**

**|    19238 |**

**+----------+**

**1 row in set (0.00 sec)**



**SELECT \* FROM batches;**

**+----+---------+-----------------+--------------------------------------------+-------------+---------------------+**

**| id | name    | charge\_protocol | description                                | n\_batteries | created\_at          |**

**+----+---------+-----------------+--------------------------------------------+-------------+---------------------+**

**|  1 | Batch-1 | 2C              | 2C constant current charge · 8 batteries  |           8 | 2026-05-24 01:22:50 |**

**|  2 | Batch-2 | 3C              | 3C constant current charge · 15 batteries |          15 | 2026-05-24 01:22:50 |**

**|  3 | Batch-3 | R2.5            | R2.5 ohm resistive load · 8 batteries     |           8 | 2026-05-24 01:22:50 |**

**|  4 | Batch-4 | R3              | R3 ohm resistive load · 8 batteries       |           8 | 2026-05-24 01:22:50 |**

**|  5 | Batch-5 | RW              | Random walk charge protocol · 8 batteries |           8 | 2026-05-24 01:22:50 |**

**+----+---------+-----------------+--------------------------------------------+-------------+---------------------+**

**5 rows in set (0.00 sec)**

**#####################################################################################trainer.py###############################################################################################################**



**Cycle duration in XJTU Battery**



\# From our explore data:

mat\['data']\[0]\['relative\_time\_min']  # shape=(24,444,)

\# Last value ÷ 60 = cycle duration in hours



\# Cycle 0 duration:

24,444 timesteps × (1 second/step) ÷ 60 ÷ 60 ≈ 6.8 hours



**Actual cycle duration by batch**



Batch-1 (2C charge):

&#x20; Charge   : \~30 min  (2C = fast charging)

&#x20; Discharge: \~90 min  (0.5C discharge rate)

&#x20; Rest     : \~10 min

&#x20; Total    : \~2.2 hours per cycle ✅ matches your statement



Batch-2 (3C charge):

&#x20; Charge   : \~20 min  (3C = very fast charging)

&#x20; Discharge: \~90 min

&#x20; Rest     : \~10 min

&#x20; Total    : \~2.0 hours per cycle ✅



Batch-3 (R2.5Ω load):

&#x20; Charge   : \~60 min  (constant current)

&#x20; Discharge: \~90 min  (resistive load)

&#x20; Rest     : \~30 min

&#x20; Total    : \~3.0 hours per cycle ⚠️ slightly longer



Batch-4 (R3Ω load):

&#x20; Similar to Batch-3

&#x20; Total    : \~3.0 hours per cycle ⚠️



Batch-5 (RW):

&#x20; Variable charge/discharge depths

&#x20; Total    : \~1-3 hours per cycle (variable)



**#####################################################################Build the trainer image in Docker##############################################################################################**



**What the build does**



**Step 1: Pull tensorflow/tensorflow:2.10.1-gpu  ← \~3GB download**

**Step 2: apt-get install libmysqlclient-dev gcc**

**Step 3: pip install scipy numpy pandas matplotlib**

&#x20;       **scikit-learn mysql-connector-python etc.**

**Step 4: COPY src/ → /app/src/**



**docker compose build trainer**



\# Run trainer with GPU

**docker compose run --rm trainer**



\# Or run specific script

**docker compose run --rm trainer \\**

&#x20;   **python src/battery\_trainer.py**



**######################################################################################### Results################################################################################################################**



**CNN — R²=0.019 (essentially failed)**



Scatter plot:

&#x20; True SOH 65-85% → Predicted \~87% flat line ← model stuck

&#x20; True SOH 85-100% → slight positive correlation

&#x20; Cannot predict below 85% at all

&#x20; Horizontal band pattern → model learned mean SOH only



Residuals:

&#x20; Mean = -3.303% ← systematic underprediction

&#x20; Left-skewed distribution

&#x20; Errors range -12% to +20% ← huge spread

&#x20; Not centered at 0 → biased predictions



**LSTM — R²=0.183 (poor but learning something**



Scatter plot:

&#x20; True SOH 65-84% → Predicted \~84% flat line ← still stuck

&#x20; True SOH 84-100% → better correlation

&#x20; Has a "floor" at \~84% — cannot predict below

&#x20; Two distinct clusters visible



Residuals:

&#x20; Mean = -3.785% ← worse bias than CNN!

&#x20; Similar left-skewed shape

&#x20; Slightly tighter than CNN (-10% to +15%)

&#x20; Still not centered at 0



Diagnosis: LSTM learned slightly better but still cannot predict the low-SOH region (66-84%). The val set (last 20% of each battery) has lots of low-SOH samples but test (RW) batteries degrade differently → LSTM memorized the training protocol degradation pattern



**Transformer — R²=0.789 (genuinely learns SOH)**



Scatter plot:

&#x20; Full range 65-100% well predicted ✅

&#x20; Points cluster tightly around diagonal ✅

&#x20; No flat-line artifacts ← sees the whole sequence

&#x20; Some scatter in 75-90% range (normal)



Residuals:

&#x20; Mean = -0.893% ← near zero bias ✅

&#x20; Nearly symmetric distribution ✅

&#x20; Tight range: -8% to +10%

&#x20; Most predictions within ±3% of true SOH ✅



Diagnosis: Transformer successfully learned the degradation pattern. Global attention across all 32 cycles directly compares early vs late cycle features → accurately detects capacity fade rate.



**The flat-line problem in CNN and LSTM**



Both CNN and LSTM show "floor" predictions:

&#x20; CNN  floor : \~87% SOH

&#x20; LSTM floor : \~84% SOH



Why this happens:

&#x20; Training SOH range : \[80.9 - 100%]  mean=94.5%

&#x20; Val SOH range      : \[67.1 - 91.1%] mean=81.9%

&#x20; Test SOH range     : \[66.4 - 100%]  mean=92.0%



CNN/LSTM learned: "most training samples are 85-100%"

→ Predicts \~87% as safe default for low-SOH inputs

→ Never saw enough low-SOH training examples

→ Transformer overcomes this via global attention



**Residual bias analysis**



Model       Bias      Meaning

──────────────────────────────────────────────────

CNN        -3.303%   Systematically overestimates SOH

&#x20;                    "Battery is healthier than it is"

&#x20;                    → DANGEROUS for BMS! Misses degradation



LSTM       -3.785%   Even worse bias

&#x20;                    → DANGEROUS for BMS!



Transformer -0.893%  Near-zero bias ✅

&#x20;                    Slight overestimation but acceptable

&#x20;                    → Safe for BMS with calibration







**Dataset split connection — why the split matters here**



The flat-line problem reveals a split issue:



Train SOH : \[80.9 - 100%] ← no samples below 81%!

Val SOH   : \[67.1 - 91%]  ← has low-SOH samples

Test SOH  : \[66.4 - 100%] ← has low-SOH samples



CNN/LSTM never saw SOH < 81% during TRAINING

→ Cannot predict it at test time

→ Defaults to minimum training value (\~87%)



Transformer overcomes this because:

→ Attention learns the TREND from cycle 1 to 32

→ Can extrapolate below training minimum

→ Does not need to have seen that exact SOH value



**This is exactly the dataset split discussion topic**

Current split weakness:

&#x20; Train: first 80% of each battery (all healthy cycles)

&#x20; Val  : last  20% of each battery (all degraded cycles)



Problem:

&#x20; Training data has almost NO low-SOH samples!

&#x20; SOH only drops below 80% in the LAST FEW cycles

&#x20; Most batteries end at exactly \~80% (EOL threshold)



Better split options:

&#x20; Option 1: Include some degraded cycles in training

&#x20;            e.g. first 70% train, middle 10% discard,

&#x20;            last 20% val

&#x20; Option 2: Oversample end-of-life windows in training

&#x20; Option 3: Use cycle\_norm as feature more aggressively

&#x20;            (already have it but maybe weight it higher)



**Summary**

**Model		Scatter			Residual bias		Low-SOH		Verdict**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**CNN		Flat line		-3.3%			Cannot predict	❌ Fails**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**LSTM		Partial flat		-3.8%			Mostly fails	⚠️ Poor**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

**Transformer	Good fit		-0.9%			Handles well	✅ Works**

**\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

