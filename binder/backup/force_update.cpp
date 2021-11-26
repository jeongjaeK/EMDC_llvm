#include <iostream>
#include <elfio/elfio.hpp>

using namespace ELFIO;

int main( int argc, char** argv ){
	if ( argc != 2 ){
		std::cout << "Usage : update-section <elf_file>" << std::endl;
		return 1;
	}

	elfio reader;

	if ( !reader.load( argv[1] ) ){
		std::cout << "Failed to load ELF file" << argv[1] << std::endl;
		return 2;
	}

	Elf_Half sec_num = reader.sections.size();
	std::cout << "Number of sections : " << sec_num << std::endl;

	section* psec;

	psec = reader.sections[17]; // __CLANG_OFFLOAD_BUNDLE__sycl-spir64_fpga
	auto next_offset = psec->get_offset() + psec->get_addr_align() ;
	psec->set_size( 0 );
	
	for(int sec_idx=18; sec_idx <= 20; ++sec_idx){
		psec = reader.sections[sec_idx]; //.eh_frame_hdr
		psec->set_offset( next_offset );
		psec->set_address( 0x400000 + psec->get_offset() );
		next_offset = psec->get_offset() + psec->get_size();
	}

	segment* pseg;
	pseg = reader.segments[2]; // LOAD
	pseg->set_memory_size( psec->get_offset() + psec->get_size() );
	pseg->set_align( 0x100000 );
	pseg = reader.segments[6]; // GNU_EH_FRAME
	pseg->set_offset( 0x36450 );
	pseg->set_virtual_address( 0x436450 );
	pseg->set_physical_address( 0x436450 );

	for( int sec_idx=21; sec_idx <= 28; ++sec_idx){
		psec = reader.sections[sec_idx];
		psec->set_offset( psec->get_offset() - 0x10000 );
		psec->set_address( psec->get_address() - 0x110000 );	
	}


	pseg = reader.segments[3]; // LOAD
	pseg->set_align( 0x100000 );
	pseg->set_offset( 0x51b08 );
	pseg->set_virtual_address( 0x551b08 );
	pseg->set_physical_address( 0x551b08 );

	pseg = reader.segments[4]; // DYNAMIC
	pseg->set_offset( 0x51de0 );
	pseg->set_virtual_address( 0x551de0 );
	pseg->set_physical_address( 0x551de0 );

	pseg = reader.segments[8]; //GNU_RELRO
	pseg->set_offset( 0x51b08 );
	pseg->set_virtual_address( 0x551b08 );
	pseg->set_physical_address( 0x551b08 );


	if ( reader.save("test") ) {
		std::cout << "Successfully saved." << std::endl;	
	}
	else{
		std::cout << "Failed to save." << std::endl;	
	}
	
	
	
	
	return 0;	
}
