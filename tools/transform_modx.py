import re
import sys

def transform_rapid_code(input_text, pc_ip="192.168.1.100", pc_port=5000):
    """
    Transforms standard Grasshopper-generated RAPID code into Hand-Eye 
    calibration code with TCP socket communication.
    """
    
    # 1. Define the variables and the custom procedure to inject
    variables_to_inject = f"""
    VAR socketdev client_socket;
    VAR string pc_ip := "{pc_ip}"; 
    VAR num pc_port := {pc_port};
"""

    custom_procedure = """
    PROC SendPoseAndCapture(robtarget p, speeddata spd)
        VAR robtarget p_curr;
        VAR string msg;
        VAR string ack;
        
        ! Move to position (always fine for calibration)
        MoveJ p, spd, fine, tool_0\\WObj:=wobjBase;
        
        WaitRob \\InPos;
        WaitTime 0.5;
        
        p_curr := CRobT(\\Tool:=tool_0 \\WObj:=wobjBase);
        
        msg := NumToStr(p_curr.trans.x, 3) + "," +
               NumToStr(p_curr.trans.y, 3) + "," +
               NumToStr(p_curr.trans.z, 3) + "," +
               NumToStr(p_curr.rot.q1, 5) + "," +
               NumToStr(p_curr.rot.q2, 5) + "," +
               NumToStr(p_curr.rot.q3, 5) + "," +
               NumToStr(p_curr.rot.q4, 5);
               
        SocketSend client_socket \\Str:=msg;
        SocketReceive client_socket \\Str:=ack;
    ENDPROC
"""

    # 2. Inject Variables before PROC main()
    output_text = re.sub(r'(\s*PROC main\(\))', variables_to_inject + r'\1', input_text)
    
    # 3. Inject SocketConnect at the start of PROC main()
    connection_code = """
        SocketCreate client_socket;
        SocketConnect client_socket, pc_ip, pc_port;
"""
    output_text = re.sub(r'(PROC main\(\)\s*(?:ConfL \\Off;\s*)?(?:ConfJ \\On;\s*)?)', r'\1' + connection_code, output_text)
    
    # 4. Inject SocketClose at the end of PROC main()
    close_code = """
        SocketSend client_socket \\Str:="DONE";
        SocketClose client_socket;
    """
    output_text = re.sub(r'(\s*ENDPROC\s*ENDMODULE)', close_code + r'\1', output_text)

    # 5. Transform MoveJ to SendPoseAndCapture
    # This regex captures the robtarget array and the speeddata, ignoring zone and tool
    # Example match: MoveJ [[x,y,z], [q1,q2,q3,q4], [...], [...]], v500, z50, tool_0\WObj:=wobjBase;
    movej_pattern = re.compile(r'MoveJ\s+(\[\[.*?\],\s*\[.*?\],\s*\[.*?\],\s*\[.*?\]\])\s*,\s*([a-zA-Z0-9_]+)\s*,.*?;')
    
    output_text = movej_pattern.sub(r'SendPoseAndCapture \1, \2;', output_text)
    
    # 6. Append the custom procedure before ENDMODULE
    output_text = re.sub(r'(\s*ENDMODULE)', custom_procedure + r'\1', output_text)
    
    return output_text

# If running directly from terminal
if __name__ == "__main__":
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="Convert standard .modx to Calibration TCP .modx")
        parser.add_argument("input_file", help="Path to input .modx file")
        parser.add_argument("output_file", help="Path to save output .modx file")
        parser.add_argument("--ip", default="192.168.1.100", help="PC IP Address")
        
        try:
            args = parser.parse_args()
            with open(args.input_file, 'r') as f:
                original = f.read()
            modified = transform_rapid_code(original, pc_ip=args.ip)
            with open(args.output_file, 'w') as f:
                f.write(modified)
            print(f"Transformed RAPID code saved to {args.output_file}")
        except SystemExit:
            pass
    else:
        # This handles the case where it's run inside Grasshopper/RhinoCode 
        # without command-line arguments.
        print("transform_modx loaded. Call transform_rapid_code(text) in your script.")
