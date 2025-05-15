
from rungp import rungp 
from functions.gppretty import gppretty
from functions.popbrowser import popbrowser
import matplotlib.pyplot as plt

gp = rungp()

gppretty(gp, 'best')

popbrowser(gp)

#popbrowser(gp, history=True)


    


