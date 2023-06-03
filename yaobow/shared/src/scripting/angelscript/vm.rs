use std::{cell::RefCell, rc::Rc};

use super::{
    global_context::ScriptGlobalContext,
    module::{ScriptFunction, ScriptModule},
};

pub struct ScriptVm {
    context: Rc<RefCell<ScriptGlobalContext>>,
    module: Option<Rc<RefCell<ScriptModule>>>,
    function_index: usize,
    pc: usize,
    stack: Vec<u8>,
    sp: usize,
    r1: u32,
    r2: u32,
}

impl ScriptVm {
    const DEFAULT_STACK_SIZE: usize = 4096;
    pub fn new(context: Rc<RefCell<ScriptGlobalContext>>) -> Self {
        Self {
            context,
            module: None,
            function_index: 0,
            pc: 0,
            stack: vec![0; Self::DEFAULT_STACK_SIZE],
            sp: Self::DEFAULT_STACK_SIZE,
            r1: 0,
            r2: 0,
        }
    }

    pub fn set_module(&mut self, module: Rc<RefCell<ScriptModule>>) {
        self.module = Some(module);
    }

    pub fn set_function(&mut self, index: usize) {
        self.function_index = index;
    }

    pub fn pop_stack_i32(&mut self) -> i32 {
        let mut ret: u32 = 0;
        self.store4(&mut ret);
        ret as i32
    }

    pub fn push_ret_i32(&mut self, ret: i32) {
        self.set4(ret as u32)
    }

    pub fn execute(&mut self) {
        if self.module.is_none() {
            return;
        }

        let module = self.module.clone().unwrap();
        let module_ref = module.borrow();
        let function = module_ref.functions[self.function_index].clone();
        let mut reg: u32 = 0;

        loop {
            let inst = self.read_inst(&function);
            macro_rules! command {
                ($cmd_name: ident $(, $param_name: ident : $param_type: ident)*) => {{
                    $(let $param_name = data_read::$param_type(&function.inst, &mut self.pc);)*
                    self.$cmd_name($($param_name ,)*);
                }};

                ($cmd_name: ident : $g_type: ident $(, $param_name: ident : $param_type: ident)*) => {{
                    $(let $param_name = data_read::$param_type(&function.inst, &mut self.pc);)*
                    self.$cmd_name::<$g_type>($($param_name)*);
                }};
            }

            match inst {
                0 => command!(pop, size: u16),
                1 => command!(push, size: u16),
                2 => command!(set4, size: u32),
                3 => self.rd4(),
                4 => command!(rdsf4, index: u16),
                5 => self.wrt4(),
                6 => self.mov4(),
                7 => command!(psf, index: u16),
                8 => command!(movsf4, index: u16),
                9 => self.swap::<u32>(),
                10 => self.store4(&mut reg),
                11 => self.recall4(reg),
                12 => command!(call, function: u32),
                13 => {
                    command!(ret, param_size: u16);
                    return;
                }
                14 => command!(jmp, offset: i32),
                15 => command!(jz, offset: i32),
                16 => command!(jnz, offset: i32),
                17 => self.tz(),
                18 => self.tnz(),
                19 => self.ts_ltz(),
                20 => self.tns_gez(),
                21 => self.tp_gtz(),
                22 => self.tnp_lez(),
                23 => self.add::<i32>(),
                24 => self.sub::<i32>(),
                25 => self.mul::<i32>(),
                26 => self.div::<i32>(0),
                27 => self.xmod::<i32>(0),
                28 => self.neg::<i32>(),
                29 => self.cmp::<i32>(),
                30 => self.inc::<i32>(1),
                31 => self.dec::<i32>(1),
                32 => self.i2f(),
                33 => self.add::<f32>(),
                34 => self.sub::<f32>(),
                35 => self.mul::<f32>(),
                36 => self.div::<f32>(0.),
                37 => self.xmod::<f32>(0.),
                38 => self.neg::<f32>(),
                39 => self.cmp::<f32>(),
                40 => self.inc::<f32>(1.),
                41 => self.dec::<f32>(1.),
                42 => self.f2i(),
                43 => self.bnot(),
                44 => self.band(),
                45 => self.bor(),
                46 => self.bxor(),
                47 => self.bsll(),
                48 => self.bsrl(),
                49 => self.bsra(),
                50 => self.ui2f(),
                51 => self.f2ui(),
                52 => self.cmp::<u32>(),
                53 => self.sb(),
                54 => self.sw(),
                55 => self.ub(),
                56 => self.uw(),
                57 => self.wrt1(),
                58 => self.wrt2(),
                59 => self.inc::<i16>(1),
                60 => self.inc::<i8>(1),
                61 => self.dec::<i16>(1),
                62 => self.dec::<i8>(1),
                63 => self.push_zero(),
                64 => command!(copy, count: u16),
                65 => command!(pga, index: i32),
                66 => command!(set8, data: u64),
                67 => self.wrt8(),
                68 => self.rd8(),
                69 => self.neg::<f64>(),
                70 => self.inc::<f64>(1.),
                71 => self.dec::<f64>(1.),
                72 => self.add::<f64>(),
                73 => self.sub::<f64>(),
                74 => self.mul::<f64>(),
                75 => self.div::<f64>(0.),
                76 => self.xmod::<f64>(0.),
                77 => self.swap::<f64>(),
                78 => self.cmp::<f64>(),
                79 => self.d2i(),
                80 => self.d2ui(),
                81 => self.d2f(),
                82 => self.x2d::<i32>(),
                83 => self.x2d::<u32>(),
                84 => self.x2d::<f32>(),
                85 => self.jmpp(),
                86 => self.sret4(),
                87 => self.sret8(),
                88 => self.rret4(),
                89 => self.rret8(),
                90 => command!(str, index: u16),
                91 => command!(js_jgez, offset: i32),
                92 => command!(jns_jlz, offset: i32),
                93 => command!(jp_jlez, offset: i32),
                94 => command!(jnp_jgz, offset: i32),
                95 => command!(cmpi: i32, rhs: i32),
                96 => command!(cmpi: u32, rhs: u32),
                97 => command!(callsys, function_index: i32),
                98 => command!(callbnd, function_index: u32),
                99 => command!(rdga4, index: i32),
                100 => command!(movga4, index: i32),
                101 => command!(addi: i32, rhs: i32),
                102 => command!(subi: i32, rhs: i32),
                103 => command!(cmpi: f32, rhs: f32),
                104 => command!(addi: f32, rhs: f32),
                105 => command!(subi: f32, rhs: f32),
                106 => command!(muli: i32, rhs: i32),
                107 => command!(muli: f32, rhs: f32),
                108 => self.suspend(),
                109 => command!(alloc, this: i32, index: i32),
                110 => unimplemented!("byte code 110 - free"),
                111 => unimplemented!("byte code 111 - loadobj"),
                112 => unimplemented!("byte code 112 - storeobj"),
                113 => unimplemented!("byte code 113 - getobj"),
                114 => unimplemented!("byte code 114 - refcpy"),
                115 => unimplemented!("byte code 115 - chkref"),
                116 => unimplemented!("byte code 116 - rd1"),
                117 => unimplemented!("byte code 117 - rd2"),
                118 => unimplemented!("byte code 118 - getobjref"),
                119 => unimplemented!("byte code 119 - getref"),
                120 => unimplemented!("byte code 120 - swap48"),
                121 => unimplemented!("byte code 121 - swap84"),
                122 => unimplemented!("byte code 122 - objtype"),
                i => unimplemented!("byte code {}", i),
            }
        }
    }

    fn read_inst(&mut self, function: &ScriptFunction) -> u8 {
        let inst = function.inst[self.pc];
        self.pc += 4;
        inst
    }

    fn pop(&mut self, size: u16) {
        self.sp += size as usize;
    }

    fn push(&mut self, size: u16) {
        self.sp -= size as usize;
    }

    fn set4(&mut self, data: u32) {
        self.sp -= 4;
        unsafe {
            self.write_stack(self.sp, data);
        }
    }

    fn rd4(&mut self) {
        unsafe {
            let pos: u32 = self.read_stack(self.sp);
            let data: u32 = self.read_stack(pos as usize);
            self.write_stack(self.sp, data);
        }
    }

    fn rdsf4(&mut self, index: u16) {
        unsafe {
            let data: u32 = self.read_stack(self.stack.len() - index as usize * 4);
            self.write_stack(self.sp, data);
        }
    }

    fn wrt4(&mut self) {
        unsafe {
            let pos: u32 = self.read_stack(self.sp);
            self.sp += 4;
            let data: u32 = self.read_stack(self.sp);
            self.write_stack(pos as usize, data);
        }
    }

    fn mov4(&mut self) {
        self.wrt4();
        self.sp += 4;
    }

    fn psf(&mut self, index: u16) {
        unsafe {
            let pos = self.stack.len() - index as usize * 4;
            self.sp -= 4;
            self.write_stack(self.sp, pos);
        }
    }

    fn movsf4(&mut self, index: u16) {
        unsafe {
            let pos = self.stack.len() - index as usize * 4;
            let data: u32 = self.read_stack(pos);
            self.write_stack(pos, data);
            self.sp += 4;
        }
    }

    fn swap<T: Copy>(&mut self) {
        unsafe {
            let size = std::mem::size_of::<T>();
            let data: T = self.read_stack(self.sp);
            let data2: T = self.read_stack(self.sp + size);
            self.write_stack(self.sp, data2);
            self.write_stack(self.sp + size, data);
        }
    }

    fn store4(&mut self, reg: &mut u32) {
        unsafe {
            let data = self.read_stack(self.sp);
            self.sp += 4;
            *reg = data;
        }
    }

    fn recall4(&mut self, reg: u32) {
        unsafe {
            self.sp -= 4;
            self.write_stack(self.sp, reg);
        }
    }

    fn call(&mut self, function: u32) {
        println!("Unimplemented: call: {}", function);
    }

    fn callbnd(&mut self, function: u32) {
        println!("Unimplemented: call: {}", function);
    }

    fn rdga4(&mut self, offset: i32) {
        println!("Unimplemented: rdga4: {}", offset);
    }

    fn callsys(&mut self, function: i32) {
        let index = -function - 1;
        let context = self.context.clone();
        context.borrow_mut().call_function(self, index as usize);
    }

    fn suspend(&mut self) {
        println!("Unimplemented: suspend");
    }

    fn alloc(&mut self, this: i32, function: i32) {
        println!("Unimplemented: call global2: {} {}", this, function);
    }

    fn ret(&mut self, param_size: u16) {
        println!("Unimplemented: ret: {}", param_size);
    }

    fn jmp(&mut self, offset: i32) {
        self.pc += offset as usize;
    }

    fn jz(&mut self, offset: i32) {
        unsafe {
            let data: i32 = self.read_stack(self.sp);
            self.sp += 4;
            if data == 0 {
                self.jmp(offset);
            }
        }
    }

    fn jnz(&mut self, offset: i32) {
        unsafe {
            let data: i32 = self.read_stack(self.sp);
            self.sp += 4;
            if data != 0 {
                self.jmp(offset);
            }
        }
    }

    fn tz(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a == 0) as i32);
    }

    fn tnz(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a != 0) as i32);
    }

    fn ts_ltz(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a < 0) as i32);
    }

    fn tns_gez(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a >= 0) as i32);
    }

    fn tp_gtz(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a > 0) as i32);
    }

    fn tnp_lez(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a <= 0) as i32);
    }

    fn add<T: Copy + std::ops::Add>(&mut self) {
        self.binary_op::<T, _, _>(|a, b| b + a)
    }

    fn sub<T: Copy + std::ops::Sub>(&mut self) {
        self.binary_op::<T, _, _>(|a, b| b - a)
    }

    fn mul<T: Copy + std::ops::Mul>(&mut self) {
        self.binary_op::<T, _, _>(|a, b| b * a)
    }

    fn div<T: Copy + std::ops::Div + PartialEq>(&mut self, zero: T) {
        unsafe {
            let data1: T = self.read_stack(self.sp);
            if data1 == zero {
                panic!("divided by zero");
            }

            self.sp += 4;
            let data2: T = self.read_stack(self.sp);
            self.write_stack(self.sp, data2 / data1);
        }
    }

    fn xmod<T: Copy + std::ops::Rem + PartialEq>(&mut self, zero: T) {
        unsafe {
            let data1: T = self.read_stack(self.sp);
            if data1 == zero {
                panic!("divided by zero");
            }

            self.sp += 4;
            let data2: T = self.read_stack(self.sp);
            self.write_stack(self.sp, data2 % data1);
        }
    }

    fn neg<T: Copy + std::ops::Neg>(&mut self) {
        self.unary_op::<T, _, _>(|a| -a);
    }

    fn cmp<T: Copy + PartialOrd>(&mut self) {
        self.binary_op::<T, _, _>(|a, b| {
            if b.gt(&a) {
                1
            } else if a.gt(&b) {
                -1
            } else {
                0
            }
        })
    }

    fn inc<T: Copy + std::ops::Add>(&mut self, one: T) {
        unsafe {
            let pos: u32 = self.read_stack(self.sp);
            let data: T = self.read_stack(pos as usize);
            self.write_stack(pos as usize, data + one);
        }
    }

    fn dec<T: Copy + std::ops::Sub>(&mut self, one: T) {
        unsafe {
            let pos: u32 = self.read_stack(self.sp);
            let data: T = self.read_stack(pos as usize);
            self.write_stack(pos as usize, data - one);
        }
    }

    fn i2f(&mut self) {
        self.unary_op::<i32, _, _>(|a| a as f32);
    }

    fn f2i(&mut self) {
        self.unary_op::<f32, _, _>(|a| a as i32);
    }

    fn bnot(&mut self) {
        self.unary_op::<u32, _, _>(|a| !a);
    }

    fn band(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| b & a)
    }

    fn bor(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| b | a)
    }

    fn bxor(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| b ^ a)
    }

    fn bsll(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| b << (a & 0xff))
    }

    fn bsrl(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| b >> (a & 0xff))
    }

    fn bsra(&mut self) {
        self.binary_op::<i32, _, _>(|a, b| b >> (a & 0xff))
    }

    fn ui2f(&mut self) {
        self.unary_op::<u32, _, _>(|a| a as f32);
    }

    fn f2ui(&mut self) {
        self.unary_op::<f32, _, _>(|a| a as u32);
    }

    fn sb(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a as i8) as i32);
    }

    fn sw(&mut self) {
        self.unary_op::<i32, _, _>(|a| (a as i16) as i32);
    }

    fn ub(&mut self) {
        self.unary_op::<u32, _, _>(|a| (a as u8) as u32);
    }

    fn uw(&mut self) {
        self.unary_op::<u32, _, _>(|a| (a as u16) as u32);
    }

    fn wrt1(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| (b & 0xFFFFFF00) + (a & 0xFF));
    }

    fn wrt2(&mut self) {
        self.binary_op::<u32, _, _>(|a, b| (b & 0xFFFF0000) + (a & 0xFFFF));
    }

    fn push_zero(&mut self) {
        self.sp -= 4;
        unsafe {
            self.write_stack(self.sp, 0);
        }
    }

    fn copy(&mut self, count: u16) {
        unsafe {
            let dst: u32 = self.read_stack(self.sp);
            self.sp += 4;
            let src: u32 = self.read_stack(self.sp);

            for i in 0..count {
                let data: u32 = self.read_stack(src as usize + i as usize);
                self.write_stack(dst as usize + i as usize, data);
            }
        }
    }

    fn set8(&mut self, data: u64) {
        unsafe {
            self.sp -= 8;
            self.write_stack(self.sp, data);
        }
    }

    fn rd8(&mut self) {
        unsafe {
            let pos: u32 = self.read_stack(self.sp);
            self.sp += 4;
            let data: u64 = self.read_stack(self.sp);
            self.write_stack(pos as usize, data);
        }
    }

    fn wrt8(&mut self) {
        unsafe {
            let pos: u32 = self.read_stack(self.sp);
            self.sp -= 4;
            let data: u64 = self.read_stack(pos as usize);
            self.write_stack(self.sp, data);
        }
    }

    fn d2i(&mut self) {
        unsafe {
            let data: f64 = self.read_stack(self.sp);
            self.sp += 4;
            self.write_stack(self.sp, data as i32);
        }
    }

    fn d2ui(&mut self) {
        unsafe {
            let data: f64 = self.read_stack(self.sp);
            self.sp += 4;
            self.write_stack(self.sp, data as u32);
        }
    }

    fn d2f(&mut self) {
        unsafe {
            let data: f64 = self.read_stack(self.sp);
            self.sp += 4;
            self.write_stack(self.sp, data as f32);
        }
    }

    fn x2d<T: Copy + std::convert::Into<f64>>(&mut self) {
        unsafe {
            let data: i32 = self.read_stack(self.sp);
            self.sp += 8;
            self.sp -= std::mem::size_of::<T>();
            self.write_stack(self.sp, data as f64);
        }
    }

    fn jmpp(&mut self) {
        unsafe {
            let data: i32 = self.read_stack(self.sp);
            self.sp += 4;
            self.pc += (8 * data) as usize;
        }
    }

    fn sret4(&mut self) {
        unsafe {
            let data: u32 = self.read_stack(self.sp);
            self.sp += 4;
            self.r1 = data;
        }
    }

    fn sret8(&mut self) {
        unsafe {
            self.r1 = self.read_stack(self.sp);
            self.sp += 4;
            self.r2 = self.read_stack(self.sp);
            self.sp += 4;
        }
    }

    fn rret4(&mut self) {
        unsafe {
            self.sp -= 4;
            self.write_stack(self.sp, self.r1);
        }
    }

    fn rret8(&mut self) {
        unsafe {
            self.sp -= 4;
            self.write_stack(self.sp, self.r2);
            self.sp -= 4;
            self.write_stack(self.sp, self.r1);
        }
    }

    fn js_jgez(&mut self, offset: i32) {
        self.j(offset, |data| data >= 0);
    }

    fn jns_jlz(&mut self, offset: i32) {
        self.j(offset, |data| data < 0);
    }

    fn jp_jlez(&mut self, offset: i32) {
        self.j(offset, |data| data <= 0);
    }

    fn jnp_jgz(&mut self, offset: i32) {
        self.j(offset, |data| data > 0);
    }

    fn cmpi<T: Copy + PartialOrd>(&mut self, rhs: T) {
        unsafe {
            let data: T = self.read_stack(self.sp);
            self.write_stack(
                self.sp,
                if rhs.gt(&data) {
                    1
                } else if data.gt(&rhs) {
                    -1
                } else {
                    0
                },
            );
        }
    }

    fn addi<T: Copy + std::ops::Add>(&mut self, rhs: T) {
        unsafe {
            let data: T = self.read_stack(self.sp);
            self.write_stack(self.sp, data + rhs);
        }
    }

    fn subi<T: Copy + std::ops::Sub>(&mut self, rhs: T) {
        unsafe {
            let data: T = self.read_stack(self.sp);
            self.write_stack(self.sp, data - rhs);
        }
    }

    fn muli<T: Copy + std::ops::Mul>(&mut self, rhs: T) {
        unsafe {
            let data: T = self.read_stack(self.sp);
            self.write_stack(self.sp, data * rhs);
        }
    }

    fn pga(&mut self, index: i32) {
        let data = if index > 0 {
            let module = self.module.as_mut().unwrap().borrow();
            module.globals[index as usize]
        } else {
            let context = self.context.borrow();
            context.get_var((-index - 1) as usize)
        };

        self.sp -= 4;

        unsafe {
            self.write_stack(self.sp, data);
        }
    }

    fn movga4(&mut self, index: i32) {
        let data: u32 = unsafe { self.read_stack(self.sp) };

        if index > 0 {
            let mut module = self.module.as_mut().unwrap().borrow_mut();
            module.globals[index as usize] = data;
        } else {
            let mut context = self.context.borrow_mut();
            context.set_var((-index - 1) as usize, data);
        };

        self.sp += 4;
    }

    fn str(&mut self, index: u16) {
        let module = self.module.as_ref().unwrap().clone();
        let module_ref = module.borrow();
        let string = &module_ref.strings[index as usize];
        unsafe {
            self.sp -= 4;
            self.write_stack(self.sp, index as u32);
            self.sp -= 4;
            self.write_stack(self.sp, string.len());
        }
    }

    #[inline]
    fn j<F: Fn(i32) -> bool>(&mut self, offset: i32, f: F) {
        unsafe {
            let data: i32 = self.read_stack(self.sp);
            if f(data) {
                self.pc += offset as usize;
            }
        }
    }

    #[inline]
    fn unary_op<T: Copy, U, F: Fn(T) -> U>(&mut self, f: F) {
        unsafe {
            let data: T = self.read_stack(self.sp);
            self.write_stack(self.sp, f(data));
        }
    }

    #[inline]
    fn binary_op<T: Copy, U, F: Fn(T, T) -> U>(&mut self, f: F) {
        unsafe {
            let data: T = self.read_stack(self.sp);
            self.sp += std::mem::size_of::<T>();
            let data2: T = self.read_stack(self.sp);
            self.sp += std::mem::size_of::<T>();
            self.sp -= std::mem::size_of::<U>();
            self.write_stack(self.sp, f(data, data2));
        }
    }

    #[inline]
    unsafe fn write_stack<T>(&mut self, pos: usize, data: T) {
        *(&mut self.stack[pos] as *mut u8 as *mut T) = data;
    }

    #[inline]
    unsafe fn read_stack<T: Copy>(&self, pos: usize) -> T {
        *(&self.stack[pos] as *const u8 as *const T)
    }
}

mod data_read {
    use byteorder::{LittleEndian, ReadBytesExt};

    pub(super) fn u16(inst: &[u8], pc: &mut usize) -> u16 {
        *pc += 2;
        (&inst[*pc - 2..*pc]).read_u16::<LittleEndian>().unwrap()
    }

    pub(super) fn i16(inst: &[u8], pc: &mut usize) -> i16 {
        *pc += 2;
        (&inst[*pc - 2..*pc]).read_i16::<LittleEndian>().unwrap()
    }

    pub(super) fn i32(inst: &[u8], pc: &mut usize) -> i32 {
        *pc += 4;
        (&inst[*pc - 4..*pc]).read_i32::<LittleEndian>().unwrap()
    }

    pub(super) fn u32(inst: &[u8], pc: &mut usize) -> u32 {
        *pc += 4;
        (&inst[*pc - 4..*pc]).read_u32::<LittleEndian>().unwrap()
    }

    pub(super) fn f32(inst: &[u8], pc: &mut usize) -> f32 {
        *pc += 4;
        (&inst[*pc - 4..*pc]).read_f32::<LittleEndian>().unwrap()
    }

    pub(super) fn u64(inst: &[u8], pc: &mut usize) -> u64 {
        *pc += 8;
        (&inst[*pc - 8..*pc]).read_u64::<LittleEndian>().unwrap()
    }
}