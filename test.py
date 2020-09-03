from subprocess import call

# test script for divide code
call(['python', './load/data_load.py',
      '--out_path', './load.bin'
      ])

call(['python', './front/front.py',
      '--data', './load.bin',
      '--out_path', './front.bin'
      ])

call(['python', './middle/middle.py',
      '--data', './front.bin',
      '--out_path', './middle.bin'
      ])

call(['python', './rear/rear.py',
      '--data', './middle.bin',
      '--out_path', './rear.bin',
      '--image', './load.bin'
      ])

