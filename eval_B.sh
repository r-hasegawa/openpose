for index in `seq -f '%03g' 220 278`
do
	CLUTTER_BACKEND=x11 ./build/examples/user_code/eval_B.bin --model_pose BODY_21A --tracking 1 --render_pose 1 --write_json /data/eval_json/Bv/$index --video /data/eval_input/Bv/$index.avi 
	CLUTTER_BACKEND=x11 ./build/examples/user_code/eval_B.bin --model_pose BODY_21A --tracking 1 --render_pose 1 --write_json /data/eval_json/Bh/$index --video /data/eval_input/Bh/$index.avi 
	CLUTTER_BACKEND=x11 ./build/examples/user_code/eval_B.bin --model_pose BODY_21A --tracking 1 --render_pose 1 --write_json /data/eval_json/Bd/$index --video /data/eval_input/Bd/$index.avi 
	done