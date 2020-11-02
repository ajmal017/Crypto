import sys
import os                                                                       
from multiprocessing import Pool     

sys.path.append('/home/keyvan_tajbakhsh/.local/lib/python3.7/site-packages')
sys.path.append('/home/keyvan_tajbakhsh/.local/lib/python3.7/site-packages')
sys.path.append('')

print (sys.path)
                                                                                
processes = ('BTC_EUR.py', 'ETH_EUR.py', 'DASH_EUR.py', 'XRP_EUR.py', 'DOT_EUR.py')                                    
                                                  
                                                                                
def run_process(process):                                                             
    os.system('python3 {}'.format(process))                                       
                                                                                
                                                                                
pool = Pool(processes=len(processes))                                                        
pool.map(run_process, processes)

