-- ==============================================================
-- RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
-- Version: 2022.1
-- Copyright (C) Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity tau_nn_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config19_s is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    p_read : IN STD_LOGIC_VECTOR (15 downto 0);
    ap_return : OUT STD_LOGIC_VECTOR (9 downto 0) );
end;


architecture behav of tau_nn_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config19_s is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_pp0_stage0 : STD_LOGIC_VECTOR (0 downto 0) := "1";
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv10_0 : STD_LOGIC_VECTOR (9 downto 0) := "0000000000";
    constant ap_const_lv32_4 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000100";
    constant ap_const_lv32_F : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001111";
    constant ap_const_lv26_3FFFFF1 : STD_LOGIC_VECTOR (25 downto 0) := "11111111111111111111110001";
    constant ap_const_lv6_0 : STD_LOGIC_VECTOR (5 downto 0) := "000000";
    constant ap_const_lv13_1 : STD_LOGIC_VECTOR (12 downto 0) := "0000000000001";
    constant ap_const_lv13_200 : STD_LOGIC_VECTOR (12 downto 0) := "0001000000000";
    constant ap_const_lv12_200 : STD_LOGIC_VECTOR (11 downto 0) := "001000000000";
    constant ap_const_lv32_C : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001100";
    constant ap_const_lv12_0 : STD_LOGIC_VECTOR (11 downto 0) := "000000000000";
    constant ap_const_lv32_A : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001010";
    constant ap_const_lv32_B : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000001011";
    constant ap_const_lv2_0 : STD_LOGIC_VECTOR (1 downto 0) := "00";
    constant ap_const_lv10_3FF : STD_LOGIC_VECTOR (9 downto 0) := "1111111111";

attribute shreg_extract : string;
    signal ap_CS_fsm : STD_LOGIC_VECTOR (0 downto 0) := "1";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_pp0_stage0 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_pp0_stage0 : signal is "none";
    signal ap_enable_reg_pp0_iter0 : STD_LOGIC;
    signal ap_enable_reg_pp0_iter1 : STD_LOGIC := '0';
    signal ap_enable_reg_pp0_iter2 : STD_LOGIC := '0';
    signal ap_idle_pp0 : STD_LOGIC;
    signal ap_block_state1_pp0_stage0_iter0 : BOOLEAN;
    signal ap_block_state2_pp0_stage0_iter1 : BOOLEAN;
    signal ap_block_state3_pp0_stage0_iter2 : BOOLEAN;
    signal ap_block_pp0_stage0_subdone : BOOLEAN;
    signal sigmoid_table_address0 : STD_LOGIC_VECTOR (9 downto 0);
    signal sigmoid_table_ce0 : STD_LOGIC;
    signal sigmoid_table_q0 : STD_LOGIC_VECTOR (9 downto 0);
    signal trunc_ln113_fu_173_p1 : STD_LOGIC_VECTOR (9 downto 0);
    signal trunc_ln113_reg_204 : STD_LOGIC_VECTOR (9 downto 0);
    signal ap_block_pp0_stage0_11001 : BOOLEAN;
    signal tmp_2_reg_209 : STD_LOGIC_VECTOR (1 downto 0);
    signal zext_ln121_fu_199_p1 : STD_LOGIC_VECTOR (63 downto 0);
    signal ap_block_pp0_stage0 : BOOLEAN;
    signal ret_V_fu_81_p4 : STD_LOGIC_VECTOR (11 downto 0);
    signal r_V_fu_73_p3 : STD_LOGIC_VECTOR (25 downto 0);
    signal trunc_ln1049_fu_101_p1 : STD_LOGIC_VECTOR (3 downto 0);
    signal tmp_fu_105_p3 : STD_LOGIC_VECTOR (9 downto 0);
    signal sext_ln1029_fu_91_p1 : STD_LOGIC_VECTOR (12 downto 0);
    signal icmp_ln1049_fu_113_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal ret_V_1_fu_119_p2 : STD_LOGIC_VECTOR (12 downto 0);
    signal p_Result_s_fu_95_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal select_ln1048_fu_125_p3 : STD_LOGIC_VECTOR (12 downto 0);
    signal ret_V_3_fu_133_p3 : STD_LOGIC_VECTOR (12 downto 0);
    signal trunc_ln116_fu_141_p1 : STD_LOGIC_VECTOR (11 downto 0);
    signal index_fu_145_p2 : STD_LOGIC_VECTOR (12 downto 0);
    signal tmp_1_fu_157_p3 : STD_LOGIC_VECTOR (0 downto 0);
    signal add_ln113_fu_151_p2 : STD_LOGIC_VECTOR (11 downto 0);
    signal index_1_fu_165_p3 : STD_LOGIC_VECTOR (11 downto 0);
    signal icmp_ln119_fu_187_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal index_2_fu_192_p3 : STD_LOGIC_VECTOR (9 downto 0);
    signal ap_NS_fsm : STD_LOGIC_VECTOR (0 downto 0);
    signal ap_idle_pp0_0to1 : STD_LOGIC;
    signal ap_reset_idle_pp0 : STD_LOGIC;
    signal ap_enable_pp0 : STD_LOGIC;
    signal ap_ce_reg : STD_LOGIC;

    component tau_nn_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config19_s_sigmoid_tabkb IS
    generic (
        DataWidth : INTEGER;
        AddressRange : INTEGER;
        AddressWidth : INTEGER );
    port (
        clk : IN STD_LOGIC;
        reset : IN STD_LOGIC;
        address0 : IN STD_LOGIC_VECTOR (9 downto 0);
        ce0 : IN STD_LOGIC;
        q0 : OUT STD_LOGIC_VECTOR (9 downto 0) );
    end component;



begin
    sigmoid_table_U : component tau_nn_sigmoid_ap_fixed_16_6_5_3_0_ap_fixed_16_6_5_3_0_sigmoid_config19_s_sigmoid_tabkb
    generic map (
        DataWidth => 10,
        AddressRange => 1024,
        AddressWidth => 10)
    port map (
        clk => ap_clk,
        reset => ap_rst,
        address0 => sigmoid_table_address0,
        ce0 => sigmoid_table_ce0,
        q0 => sigmoid_table_q0);





    ap_CS_fsm_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
            else
                ap_CS_fsm <= ap_NS_fsm;
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter1_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter1 <= ap_const_logic_0;
            else
                if (((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
                    ap_enable_reg_pp0_iter1 <= ap_start;
                end if; 
            end if;
        end if;
    end process;


    ap_enable_reg_pp0_iter2_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_enable_reg_pp0_iter2 <= ap_const_logic_0;
            else
                if ((ap_const_boolean_0 = ap_block_pp0_stage0_subdone)) then 
                    ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
                end if; 
            end if;
        end if;
    end process;

    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then
                tmp_2_reg_209 <= index_1_fu_165_p3(11 downto 10);
                trunc_ln113_reg_204 <= trunc_ln113_fu_173_p1;
            end if;
        end if;
    end process;

    ap_NS_fsm_assign_proc : process (ap_CS_fsm, ap_block_pp0_stage0_subdone, ap_reset_idle_pp0)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_pp0_stage0 => 
                ap_NS_fsm <= ap_ST_fsm_pp0_stage0;
            when others =>  
                ap_NS_fsm <= "X";
        end case;
    end process;
    add_ln113_fu_151_p2 <= std_logic_vector(unsigned(trunc_ln116_fu_141_p1) + unsigned(ap_const_lv12_200));
    ap_CS_fsm_pp0_stage0 <= ap_CS_fsm(0);
        ap_block_pp0_stage0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_pp0_stage0_11001 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_pp0_stage0_subdone <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state1_pp0_stage0_iter0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state2_pp0_stage0_iter1 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state3_pp0_stage0_iter2 <= not((ap_const_boolean_1 = ap_const_boolean_1));

    ap_done_assign_proc : process(ap_enable_reg_pp0_iter2, ap_block_pp0_stage0_subdone)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (ap_enable_reg_pp0_iter2 = ap_const_logic_1))) then 
            ap_done <= ap_const_logic_1;
        else 
            ap_done <= ap_const_logic_0;
        end if; 
    end process;

    ap_enable_pp0 <= (ap_idle_pp0 xor ap_const_logic_1);
    ap_enable_reg_pp0_iter0 <= ap_start;

    ap_idle_assign_proc : process(ap_start, ap_CS_fsm_pp0_stage0, ap_idle_pp0)
    begin
        if (((ap_start = ap_const_logic_0) and (ap_idle_pp0 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_pp0_assign_proc : process(ap_enable_reg_pp0_iter0, ap_enable_reg_pp0_iter1, ap_enable_reg_pp0_iter2)
    begin
        if (((ap_enable_reg_pp0_iter2 = ap_const_logic_0) and (ap_enable_reg_pp0_iter1 = ap_const_logic_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_0))) then 
            ap_idle_pp0 <= ap_const_logic_1;
        else 
            ap_idle_pp0 <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_pp0_0to1_assign_proc : process(ap_enable_reg_pp0_iter0, ap_enable_reg_pp0_iter1)
    begin
        if (((ap_enable_reg_pp0_iter1 = ap_const_logic_0) and (ap_enable_reg_pp0_iter0 = ap_const_logic_0))) then 
            ap_idle_pp0_0to1 <= ap_const_logic_1;
        else 
            ap_idle_pp0_0to1 <= ap_const_logic_0;
        end if; 
    end process;


    ap_ready_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter0, ap_block_pp0_stage0_subdone)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_subdone) and (ap_enable_reg_pp0_iter0 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            ap_ready <= ap_const_logic_1;
        else 
            ap_ready <= ap_const_logic_0;
        end if; 
    end process;


    ap_reset_idle_pp0_assign_proc : process(ap_start, ap_idle_pp0_0to1)
    begin
        if (((ap_start = ap_const_logic_0) and (ap_idle_pp0_0to1 = ap_const_logic_1))) then 
            ap_reset_idle_pp0 <= ap_const_logic_1;
        else 
            ap_reset_idle_pp0 <= ap_const_logic_0;
        end if; 
    end process;

    ap_return <= sigmoid_table_q0;
    icmp_ln1049_fu_113_p2 <= "1" when (tmp_fu_105_p3 = ap_const_lv10_0) else "0";
    icmp_ln119_fu_187_p2 <= "0" when (tmp_2_reg_209 = ap_const_lv2_0) else "1";
    index_1_fu_165_p3 <= 
        ap_const_lv12_0 when (tmp_1_fu_157_p3(0) = '1') else 
        add_ln113_fu_151_p2;
    index_2_fu_192_p3 <= 
        ap_const_lv10_3FF when (icmp_ln119_fu_187_p2(0) = '1') else 
        trunc_ln113_reg_204;
    index_fu_145_p2 <= std_logic_vector(unsigned(ret_V_3_fu_133_p3) + unsigned(ap_const_lv13_200));
    p_Result_s_fu_95_p2 <= "1" when (signed(r_V_fu_73_p3) < signed(ap_const_lv26_3FFFFF1)) else "0";
    r_V_fu_73_p3 <= (p_read & ap_const_lv10_0);
    ret_V_1_fu_119_p2 <= std_logic_vector(signed(sext_ln1029_fu_91_p1) + signed(ap_const_lv13_1));
    ret_V_3_fu_133_p3 <= 
        select_ln1048_fu_125_p3 when (p_Result_s_fu_95_p2(0) = '1') else 
        sext_ln1029_fu_91_p1;
    ret_V_fu_81_p4 <= p_read(15 downto 4);
    select_ln1048_fu_125_p3 <= 
        sext_ln1029_fu_91_p1 when (icmp_ln1049_fu_113_p2(0) = '1') else 
        ret_V_1_fu_119_p2;
        sext_ln1029_fu_91_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(ret_V_fu_81_p4),13));

    sigmoid_table_address0 <= zext_ln121_fu_199_p1(10 - 1 downto 0);

    sigmoid_table_ce0_assign_proc : process(ap_CS_fsm_pp0_stage0, ap_enable_reg_pp0_iter1, ap_block_pp0_stage0_11001)
    begin
        if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_enable_reg_pp0_iter1 = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_pp0_stage0))) then 
            sigmoid_table_ce0 <= ap_const_logic_1;
        else 
            sigmoid_table_ce0 <= ap_const_logic_0;
        end if; 
    end process;

    tmp_1_fu_157_p3 <= index_fu_145_p2(12 downto 12);
    tmp_fu_105_p3 <= (trunc_ln1049_fu_101_p1 & ap_const_lv6_0);
    trunc_ln1049_fu_101_p1 <= p_read(4 - 1 downto 0);
    trunc_ln113_fu_173_p1 <= index_1_fu_165_p3(10 - 1 downto 0);
    trunc_ln116_fu_141_p1 <= ret_V_3_fu_133_p3(12 - 1 downto 0);
    zext_ln121_fu_199_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(index_2_fu_192_p3),64));
end behav;