for index in `seq -f '%03g' 1 278`
do
	./build/examples/user_code/eval_B.bin --model_pose BODY_21A --tracking 1 --render_pose 1 --write_json /data/eval_json/Bv/$index --video /data/eval_input/Bv/$index.avi
	./build/examples/user_code/eval_B.bin --model_pose BODY_21A --tracking 1 --render_pose 1 --write_json /data/eval_json/Bh/$index --video /data/eval_input/Bh/$index.avi
	./build/examples/user_code/eval_B.bin --model_pose BODY_21A --tracking 1 --render_pose 1 --write_json /data/eval_json/Bd/$index --video /data/eval_input/Bd/$index.avi
	done