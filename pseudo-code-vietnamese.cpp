function train()
{
	Load arguments (args) // Tải các tham số
	Load file weight // Tải trọng số 
	Load reference images and source images for training // Tải ảnh nội dung và ảnh phong cách để huấn luyện
	Load images for valuation // Tải ảnh để đánh giá
	For iteration from resume_iteration to total_iteration:
		Load image:
			- x_src: ảnh nội dung
			- x_ref_1: ảnh phong cách 1
			- x_ref_2: ảnh phong cách 2
			- y_org: domain ảnh nội dung
			- y_trg: domain ảnh phong cách
		- discriminator_loss(x_src, x_ref_1, y_trg) // Tính toán độ lỗi chân thật từ ảnh gốc và ảnh phong cách 1
		- update discriminator by discriminator loss // Cập nhật trọng số cho mạng discriminator từ độ lỗi chân thật
		- generator_loss(args, x_src, x_ref_2, y_trg) // Tính toán độ lỗi tổng hợp
		- update generator by generator loss // Cập nhật trọng số cho mạng generator từ độ lỗi tổng hợp
}

function discriminator_loss(x_src, x_ref_1, y_trg, y_org){
	out = discriminator(x_src, y_org)
		loss_real = adv_loss(out, 1) // Tính độ lỗi chân thật từ ảnh nội dung và domain của ảnh nội dung
	loss_reg = r1_reg(out, x_src)

		s_trg = style_encoder(x_ref_1, y_trg) // Tính mã phong cách từ ảnh phong cách 1 và domain của ảnh phong cách
	x_fake = generator(x_src, s_trg)		  // x_fake là ảnh tạo sinh từ ảnh gốc và mã phong cách tính ở trên (s_trg)
	loss_fake = discriminator(x_fake, y_trg)  // Tính độ lỗi chân thật từ ảnh tạo sinh và domain của ảnh phong cách

	loss = loss_real + loss_fake + args.lambda_reg * loss_reg // Tính tổng độ lỗi
									   return loss}

func generator_loss(args, x_src, x_ref_1, x_ref_2, y_org, y_trg){
	// Tính độ lỗi chân thật
	s_trg = style_encoder(x_ref_1, y_trg) // Tính mã phong cách từ ảnh phong cách 1 và domain của ảnh phong cách
	x_fake = generator(x_src, s_trg)	  // x_fake là ảnh tạo sinh từ ảnh gốc và mã phong cách tính ở trên (s_trg)
	out = discriminator(x_src, y_org)
		loss_adv = adv_loss(out, 1) // Tính độ lỗi chân thật từ ảnh tạo sinh (x_fake) và domain của ảnh phong cách

	// Tính độ lỗi tái tạo phong cách
	s_pred = style_encoder(x_fake, y_trg)  // Tính mã phong cách từ ảnh tạo sinh (x_fake) và domain của ảnh phong cách
	loss_style = mean(abs(s_pred - s_trg)) // Tính độ lỗi tái tạo phong cách từ
										   // mã phong cách từ ảnh phong cách 1 (s_trg)
										   // và mã phong cách từ ảnh tạo sinh (x_fake) (s_pred)

	// Tính độ lỗi đa dạng phong cách
	s_trg2 = style_encoder(x_ref_2, y_trg) // Tính mã phong cách từ ảnh tạo sinh 2 và domain của ảnh phong cách
	x_fake2 = generator(x_src, s_trg2)	   // x_fake2 là ảnh tạo sinh từ ảnh gốc và phong cách tính ở trên (s_trg2)
	loss_ds = mean(abs(x_fake - x_fake2))  // Tính độ lỗi đa dạng phong cách từ
										   // ảnh tạo sinh x_fake và ảnh tạo sinh x_fake2

	// Tính độ lõi bảo toàn nội dung
	s_org = style_encoder(x_src, x_trg) // Tính mã phong cách từ ảnh gốc và domain của ảnh nội dung
	x_rec = generator(x_fake, s_org)	// x_rec là ảnh tạo sinh từ ảnh tạo sinh (x_fake) và phong cách của ảnh nội dung
	loss_cyc = mean(abs(x_rec, x_src))	// Tính độ lỗi bảo toàn nội dung từ
										// ảnh tạo sinh (x_rec) và ảnh nội dung ban đầu

	loss = loss_adv + args.lambda_sty * loss_sty - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc // Tính độ lỗi tổng hợp từ 4 độ lỗi trên
																				  return loss}

function discriminator(x, y){// x: ảnh; y domain
							 -}

function style_encoder(x, y){// x: ảnh; y: domain
	-Trích xuất phong cách từ ảnh theo domain
	- Trả về mã phong cách}

function generator(x, y)
{ // x: ảnh; y: mã phong cách
	x = from_rgb(x)
	// Đựa ảnh qua mã phong cách
	for block in encode:
		x = block(x)
	for block in decode:
		x = block(x, s)
	return self.to_rgb(x) // Trả về ảnh tạo sinh từ ảnh x và mã phong cách y
}
