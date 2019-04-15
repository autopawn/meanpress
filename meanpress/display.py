import os

NAMES = ["D","B","K"]

def display_channel_comp(original,arrays,means,sizes,display_base_fname=None,display_channel_name="?"):
    import matplotlib.pyplot as plt

    if display_base_fname is not None:
        basefname,extension = os.path.splitext(display_base_fname)

    # Channel
    plt.title("Channel %s"%(display_channel_name))
    plt.imshow(original,cmap='Greys',vmin=0,vmax=255)
    if display_base_fname is not None:
        plt.savefig(basefname+"_cha%s.png"%(display_channel_name))
    else:
        plt.show()

    for i in range(len(arrays)):
        # Means
        plt.title("Matrix $%s^{(%d)}$ for channel %s"%("P",i+1,display_channel_name))
        plt.imshow(means[i],cmap='Greys',vmin=0,vmax=255)
        if display_base_fname is not None:
            plt.savefig(basefname+"_cha%s_step%03d_%s.png"%(display_channel_name,i+1,"P"))
        else:
            plt.show()
        # Matrix
        for j in range(len(arrays[i])):
            plt.title("Matrix $%s^{(%d)}$ for channel %s, reduced to %d bytes (%f bits per pixel)"%(
                NAMES[j],i+1,display_channel_name,sizes[i][j],8*sizes[i][j]/float(arrays[i][j].size)))
            plt.imshow(arrays[i][j],cmap='Greys',vmin=0)
            if display_base_fname is not None:
                plt.savefig(basefname+"_cha%s_step%03d_%s.png"%(display_channel_name,i+1,NAMES[j]))
            else:
                plt.show()
