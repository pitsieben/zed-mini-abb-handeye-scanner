# XML template for KUKA Ethernet KRL (EKI) communication with Robot 3D Scanning Studio
EKI_XML_TEMPLATE = """<ETHERNETKRL>
  <CONFIGURATION>
    <EXTERNAL>
      <IP>{pc_ip}</IP>
      <PORT>{pc_port}</PORT>
    </EXTERNAL>
  </CONFIGURATION>
  <RECEIVE>
    <RAW>
      <ELEMENT Tag="ACK" Type="STRING" Size="32" SetFlag="1" />
    </RAW>
  </RECEIVE>
  <SEND>
    <RAW>
      <ELEMENT Tag="Buffer" Type="STRING" Size="128" />
    </RAW>
  </SEND>
</ETHERNETKRL>
"""

import os
import re
import sys

def transform_krl_code(input_text, pc_ip="192.168.1.100", pc_port=5000, channel_name="ScannerPC"):
    """
    Transforms standard Grasshopper/KUKA|prc-generated KRL (.src) code into 
    Robot 3D Scanning Studio code with Ethernet KRL (EKI) socket communication.
    
    Coordinates are sent as X,Y,Z,A,B,C comma-separated string which the web studio
    automatically detects as Euler angles and converts to Quaternions.
    """
    declarations = """
  ;FOLD SCANNER TCP VARIABLES
  DECL EKI_STATUS RET
  ;ENDFOLD
"""

    connection_code = f"""
  ;FOLD CONNECT TO SCANNER PC
  RET = EKI_Init("{channel_name}")
  RET = EKI_Open("{channel_name}")
  WAIT SEC 0.5
  ;ENDFOLD
"""

    close_code = f"""
  ;FOLD CLOSE SCANNER PC CONNECTION
  RET = EKI_Send("{channel_name}", "DONE")
  WAIT SEC 0.5
  RET = EKI_Close("{channel_name}")
  RET = EKI_Clear("{channel_name}")
  ;ENDFOLD
"""

    custom_procedure = f"""
;FOLD SendPoseAndCapture Procedure
DEF SendPoseAndCapture(target_pos:IN)
  DECL POS target_pos
  DECL POS current_pos
  DECL EKI_STATUS RET_LOCAL
  DECL CHAR send_buf[128]
  DECL CHAR recv_buf[32]
  INT offset

  ; Fine exact stop point for camera stability
  PTP target_pos
  WAIT SEC 0.5

  current_pos = $POS_ACT

  ; Format and transmit X,Y,Z,A,B,C
  offset = 0
  SWRITE(send_buf[], RET_LOCAL, offset, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", current_pos.X, current_pos.Y, current_pos.Z, current_pos.A, current_pos.B, current_pos.C)
  RET_LOCAL = EKI_Send("{channel_name}", send_buf[])

  ; Wait for ACK flag from Python server (flag 1 set by EKI XML)
  WAIT FOR $FLAG[1] TIMEOUT=15.0
  IF $FLAG[1] THEN
    $FLAG[1] = FALSE
    RET_LOCAL = EKI_GetString("{channel_name}", "ACK", recv_buf[])
  ENDIF
END
;ENDFOLD
"""

    output = input_text

    # 1. Inject Declarations before INI
    output = re.sub(r'(;FOLD INI)', declarations + r'\n\1', output, count=1)

    # 2. Inject Connection after INI
    output = re.sub(r'(;ENDFOLD\s*\(INI\))', r'\1\n' + connection_code, output, count=1)

    # 3. Replace PTP motion AND strip trailing C_PTP / approximation flags
    output = re.sub(
        r'PTP\s+(\{[^}]*X[^}]*\})\s*(?:C_PTP|C_DIS)?',
        r'SendPoseAndCapture(\1)',
        output
    )

    # 4. Inject Close Code right before main routine END
    output = re.sub(r'(\n\s*END\s*)$', r'\n' + close_code + r'\1', output.rstrip())

    # 5. Append Procedure at bottom
    output = output.strip() + "\n\n" + custom_procedure

    return output


# --- Grasshopper Execution Pattern ---
# If imported or pasted directly in a Grasshopper GhPython / RhinoCode component:
if __name__ != "__main__":
    try:
        # Check if variables exist in Grasshopper global scope
        _input_code = code if 'code' in globals() else None
        _pc_ip = pc_ip if 'pc_ip' in globals() else "192.168.1.100"
        _pc_port = pc_port if 'pc_port' in globals() else 5000
        _channel = eki_channel if 'eki_channel' in globals() else "ScannerPC"
        
        if _input_code:
            code_text = "\n".join(_input_code) if isinstance(_input_code, list) else _input_code
            modified_code = transform_krl_code(code_text, _pc_ip, _pc_port, _channel)
            
            # Optional file saving directly from Grasshopper
            if 'write' in globals() and write:
                _folder = folder if 'folder' in globals() else "."
                _folder_name = folder_name if 'folder_name' in globals() else "kuka_output"
                full_path = os.path.join(_folder, _folder_name)
                os.makedirs(full_path, exist_ok=True)
                
                # Write .src file
                src_path = os.path.join(full_path, "HandtoEye_KUKA.src")
                with open(src_path, "w") as f:
                    f.write(modified_code)
                    
                # Write matching EKI XML config
                xml_path = os.path.join(full_path, f"{_channel}.xml")
                with open(xml_path, "w") as f:
                    f.write(EKI_XML_TEMPLATE.format(pc_ip=_pc_ip, pc_port=_pc_port))
    except Exception as e:
        print(f"Error in Grasshopper execution: {e}")

# --- Command Line Execution ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description="Convert standard KRL code to Scanner TCP KRL code.")
        parser.add_argument("input_file", help="Path to input .src file")
        parser.add_argument("output_file", help="Path to output .src file")
        parser.add_argument("--ip", default="192.168.1.100", help="PC IP Address (default: 192.168.1.100)")
        parser.add_argument("--port", type=int, default=5000, help="PC TCP Port (default: 5000)")
        parser.add_argument("--channel", default="ScannerPC", help="EKI Channel name (default: ScannerPC)")
        
        args = parser.parse_args()
        with open(args.input_file, 'r') as f:
            raw_krl = f.read()
            
        transformed = transform_krl_code(raw_krl, pc_ip=args.ip, pc_port=args.port, channel_name=args.channel)
        with open(args.output_file, 'w') as f:
            f.write(transformed)
        print(f"[OK] Saved transformed KRL to: {args.output_file}")
    else:
        print("transform_krl.py loaded. Call transform_krl_code(text) or run with arguments.")
