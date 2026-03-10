import matplotlib.pyplot as pl

if __name__=="__main__":
    #initialize figure
    fig=pl.figure(figsize=[8,8])
    
    #create axes instances for plotting
    ax_cmp=fig.add_axes([0.3, 0.35,0.5,0.5])
    #pcolormesh
    #x,y,2dval,cmap='vindis'
    #ax_yrl=ax_cmp.twiny()
    #ax_yrl.set_position([0.1,0.35,0.2,0.5])
    ax_avg_rainyearly=fig.add_axes([0.1,0.35,0.2,0.5])
    ax_avg_rainmonthly=fig.add_axes([0.3,0.1,0.5,0.25])
    #show results
    pl.show()
    plsavefig("./aimr.png")
