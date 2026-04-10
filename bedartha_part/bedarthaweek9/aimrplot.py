import pandas as pd
import matplotlib.pyplot as pl

if __name__=="__main__":
    df=pd.read_csv("all-india-monthly-rainfall.csv",index_col=0)

    years=df.index
    months=df.columns
    aimr=df.to_numpy()
    print(aimr.shape,years.shape,months.shape)

    #initialize figure
    fig=pl.figure(figsize=[8,8])
    
    #create axes instances for plotting
    ax_cmp=fig.add_axes([0.3, 0.35,0.5,0.5])
    ax_cb=fig.add_axes([0.85,0.35,0.05,0.5])
    ax_yearly=fig.add_axes([0.1,0.35,0.2,0.5])
    ax_monthly=fig.add_axes([0.3,0.1,0.5,0.25])    
    #pcolormesh
    cm=ax_cmp.pcolormesh(months,years,aimr,cmap='viridis')
    cb=pl.colorbar(cm,cax=ax_cb)
    
    ax_yearly.plot(aimr,years,color="gray",lw=0.75,alpha=0.65)
    ax_yearly.plot(aimr.mean(axis=1),years,color='r',lw=1.5,alpha=1)
    
    ax_monthly.bar(x=months,height=aimr.mean(axis=0),color="SteelBlue")

    ylims=ax_monthly.get_ylim()
    ax_monthly.set_ylim(ylims[::-1])


    #x,y,2dval,cmap='viridis'
    #ax_yrl=ax_cmp.twiny()
    #ax_yrl.set_position([0.1,0.35,0.2,0.5])

    #show results
    pl.show()
    #plsavefig("./aimr.png")
