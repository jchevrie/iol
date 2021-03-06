#!/usr/local/bin/lua

require("yarp")
require("rfsm")
--require("iol_funcs")

yarp.Network()

-------
shouldExit = false

-- load state machine model and initalize it
rf = yarp.ResourceFinder()
rf:setDefaultContext("iol/lua")
rf:configure(arg)
fsm_file = rf:findFile("iol_root_fsm_mobile.lua")
fsm_model = rfsm.load(fsm_file)
fsm = rfsm.init(fsm_model)
rfsm.run(fsm)

repeat
    rfsm.run(fsm)
    yarp.delay(0.1)
until shouldExit ~= false

print("finishing")
-- Deinitialize yarp network
yarp.Network_fini()
