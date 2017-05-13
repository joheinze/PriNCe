import prince.intcs as pcs
import matplotlib.pyplot as plt
import numpy as np

csi = pcs.CrossSectionInterpolator([(0., pcs.NeucosmaFileInterface, ()),
                                    (0.8, pcs.SophiaSuperposition, ())])

egrid = np.logspace(-1,5,200)
plt.semilogx(egrid, csi.nonel_intp[101](egrid))
plt.semilogx(egrid, csi.nonel_intp[1206](egrid))
plt.semilogx(egrid, csi.nonel_intp[5626](egrid))

plt.show()