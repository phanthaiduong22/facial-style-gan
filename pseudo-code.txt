function train(){
	Load arguments (args)
	Load file weight
	Load reference images and source images for training
	Load images for valuation
	For iteration from resume_iteration to total_iteration:
		Load image:
			- x_src: src image
			- x_ref_1: first ref image
			- x_ref_2: second ref image
			- y_trg: ref image's domain
			- y_org: src image's domain
		- discriminator_loss(x_src, x_ref_1, y_trg) // Calculate discriminator loss by first reference image and src image
		- update discriminator by discriminator loss
		- generator_loss(args, x_src, x_ref_2, y_trg) // Calculate generator loss by second reference image and src image
		- update generator by generator loss
}

function discriminator_loss(x_src, x_ref_1, y_trg){ // x_src: src image, x ref: ref image, y_trg: reference domain
	loss_real = discriminator(x_src, y_org)

	s_trg = style_encoder(x_ref_1, y_trg) // calculate style encode by first ref image and ref domain
	x_fake = generator(x_src, s_trg) // x_fake is fake image which contain reference image's style and content image
	loss_fake = discriminator(x_fake, y_trg) // calculate discriminator loss

	loss = loss_real + loss_fake// final loss
	return loss
}

func generator_loss(args, x_src, x_ref_1, x_ref_2, y_org, y_trg){ // x_src: src image, x_ref_1: first ref image, x_ref_2: second ref image, 
															// y_trg: ref image's domain, y_org: src image's domain
	// Calculate style loss
	s_trg = style_encoder(x_ref_1, y_trg) // calculate style encode by first ref image and ref domain
	x_fake = generator(x_src, s_trg) // x_fake is fake image which contain reference image's style and content image
	s_pred = style_encoder(x_fake, y_trg) // style encode of fake image above and ref domain
	loss_style = mean(abs(s_pred - s_trg)) // calculate style loss

	// Calculate diversity sensitive loss
	s_trg2 = style_encoder(x_ref_2, y_trg) // calculate style encode by second ref image and ref domain
	x_fake2 = generator(x_src, s_trg2) // x_fake2 is fake image 2 which contain reference 2 image's style and content image
	loss_ds = mean(abs(x_fake - x_fake2)) // calculate diversity sensitive loss from first fake image and second fake image

	// Calculate cycle-consistency loss
	s_org = style_encoder(x_src, x_trg) // calculate style encode by src image and src domain
	x_rec = generator(x_fake, s_org) // x_rec is fake image which contain src image's style and x_fake image
	loss_cyc = mean(abs(x_rec, x_src)) // calculate diversity sensitive loss from src image and x_rec fake image

	loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc // final loss, lambda is passing from arguments
	return loss
}


function style_encoder(x, y){ // x: image, y: domain
	h = self.shared(x)
	h = h.view(h.size(0), -1)
	out = []
	for layer in self.unshared:
		out += [layer(h)]
	out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
	idx = torch.LongTensor(range(y.size(0))).to(y.device)
	s = out[idx, y]  # (batch, style_dim)
	return s // style_encode of image and domain
}

function generator(x, y){ // x: image, y: style 
	x = self.from_rgb(x)
	cache = {}
	for block in self.encode:
		x = block(x)
	for block in self.decode:
		x = block(x, s)
	return self.to_rgb(x) // return image ge
}