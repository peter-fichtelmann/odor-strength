dot_correction = {
    # from odor strength data
   'butylated hydroxyanisole': ['COc1ccc(O)c(C(C)(C)C)c1', 'COc1ccc(O)cc1C(C)(C)C'],
#    'methyl ionone terpenes': ['CC(=O)C(C)=CC=C(C)CCC=C(C)C','CCC(=O)C=CC=C(C)CCC=C(C)C'],
    'peg-8 distearate': ['CCCCCCCCCCCCCCCCC(=O)OCCOC(=O)CCCCCCCCCCCCCCCC'],
    'isocyclocitral (IFF)': ['CC1=CC(C)C(C)C(C=O)C1', 'CC1=CC(C)C(C=O)C(C)C1'],
    'santalyl acetate': ['C=C1[C@@H]2CC[C@@H](C2)[C@@]1(C)CC/C=C(/C)COC(C)=O', 'CC(=O)OC/C(C)=C\CC[C@]1(C)C2CC3[C@H](C2)C31C'],
    'propylene glycol stearate': ['CCCCCCCCCCCCCCCCCC(=O)OC(C)CO', 'CCCCCCCCCCCCCCCCCC(=O)OC(C)CO'], # after cas mono stearate
    'CID 518601': ['C=C(C)C1C2OC(=O)C1C1(O)CC3OC34C(=O)OC2C14C', 'CC(C)(O)C1C2OC(=O)C1C1(O)CC3OC34C(=O)OC2C14C'],
    'Picrotoxin, powder': ['C=C(C)[C@@H]1[C@@H]2OC(=O)[C@H]1[C@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2[C@]14C', 'CC(C)(O)[C@@H]1[C@@H]2OC(=O)[C@H]1[C@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2[C@]14C'],
    'Oriental berry': ['C=C(C)[C@H]1[C@@H]2OC(=O)[C@H]1[C@]1(O)C[C@H]3OC34C(=O)O[C@H]2[C@@]41C', 'CC(C)(O)[C@H]1[C@@H]2OC(=O)[C@H]1[C@]1(O)C[C@H]3OC34C(=O)O[C@H]2[C@@]41C'],
    'Abamectin': ['CCC(C)[C@H]1O[C@]2(C=C[C@@H]1C)C[C@@H]1C[C@@H](C/C=C(\C)[C@@H](O[C@H]3C[C@H](OC)[C@@H](O[C@H]4C[C@H](OC)[C@@H](O)[C@H](C)O4)[C@H](C)O3)[C@@H](C)/C=C/C=C3\CO[C@@H]4[C@H](O)C(C)=C[C@@H](C(=O)O1)[C@]34O)O2', 'CO[C@H]1C[C@H](O[C@H]2[C@H](C)O[C@@H](O[C@@H]3/C(C)=C/C[C@@H]4C[C@@H](C[C@]5(C=C[C@H](C)[C@@H](C(C)C)O5)O4)OC(=O)[C@@H]4C=C(C)[C@@H](O)[C@H]5OC/C(=C\C=C\[C@@H]3C)[C@]54O)C[C@@H]2OC)O[C@@H](C)[C@@H]1O'],
    '(1S,3R,5S,8S,9S,12R,13S,14S)-1-Hydroxy-14-(2-hydroxypropan-2-yl)-13-methyl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione;(1S,3R,5S,8S,9S,12R,13S,14R)-1-hydroxy-13-methyl-14-prop-1-en-2-yl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione': ['C=C(C)[C@@H]1[C@@H]2OC(=O)[C@H]1[C@@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2[C@]41C', 'CC(C)(O)[C@@H]1[C@@H]2OC(=O)[C@H]1[C@@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2[C@]41C'],
    '(1R,3R,5S,8S,9R,12S,13R,14R)-1-hydroxy-14-(2-hydroxypropan-2-yl)-13-methyl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione;1-hydroxy-13-methyl-14-prop-1-en-2-yl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione': ['C=C(C)C1C2OC(=O)C1C1(O)CC3OC34C(=O)OC2C14C', 'CC(C)(O)[C@H]1[C@H]2OC(=O)[C@@H]1[C@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2[C@]14C'],
    '(1R,5S,13R)-1-hydroxy-14-(2-hydroxypropan-2-yl)-13-methyl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione;(1R,5S,13R)-1-hydroxy-13-methyl-14-prop-1-en-2-yl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione': ['C=C(C)C1C2OC(=O)C1[C@]1(O)CC3O[C@]34C(=O)OC2[C@]14C', 'CC(C)(O)C1C2OC(=O)C1[C@]1(O)CC3O[C@]34C(=O)OC2[C@]14C'],
    '(1R,3R,5S,8S,9S,12R,14S)-1-hydroxy-14-(2-hydroxypropan-2-yl)-13-methyl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione;(1R,3R,5S,8S,9S,12R,14R)-1-hydroxy-13-methyl-14-prop-1-en-2-yl-4,7,10-trioxapentacyclo[6.4.1.19,12.03,5.05,13]tetradecane-6,11-dione': ['C=C(C)[C@@H]1[C@@H]2OC(=O)[C@H]1[C@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2C14C', 'CC(C)(O)[C@@H]1[C@@H]2OC(=O)[C@H]1[C@]1(O)C[C@H]3O[C@]34C(=O)O[C@H]2C14C'],
    'Xylidine': ['Cc1cc(C)cc(N)c1', 'Cc1ccc(C)c(N)c1', 'Cc1ccc(N)c(C)c1', 'Cc1ccc(N)cc1C', 'Cc1cccc(C)c1N', 'Cc1cccc(N)c1C'],
    'amber furan': ['CCC23CCC1C(C)(C)CCCC1(C)C2CCO3'], # reaction product
}