function train()
{
	Load arguments (args) 
	Load file weight
	Load reference images and source images for training
	Load images for valuation
	For iteration from resume_iteration to total_iteration:
		Load image:
			- x_src: content image
			- x_ref_1: first reference image
			- x_ref_2: second reference image
			- y_org: domain of content image
			- y_trg: domain of reference image
		- discriminator_loss(x_src, x_ref_1, y_trg) // calculate discriminator loss by first reference image and src image
		- update discriminator by discriminator loss // update for discriminator by discriminator loss
		- generator_loss(args, x_src, x_ref_2, y_trg) // calculate total loss by reference images and src image
		- update generator by generator loss // update for generator by total loss
}

function discriminator_loss(x_src, x_ref_1, y_trg, y_org){
	out = discriminator(x_src, y_org)
		loss_real = adv_loss(out, 1) // calculate discriminator loss
	loss_reg = r1_reg(out, x_src)	 // calculate discriminator loss

	s_trg = style_encoder(x_ref_1, y_trg)	 // calculate style encode by first reference image and reference domain
	x_fake = generator(x_src, s_trg)		 // x_fake is generated image which contain reference image's style and content image
	loss_fake = discriminator(x_fake, y_trg) // calculate discriminator loss by generated image (x_fake) and reference domain

	loss = loss_real + loss_fake + args.lambda_reg * loss_reg // calculate total loss
									   return loss}

func generator_loss(args, x_src, x_ref_1, x_ref_2, y_org, y_trg){
	// calculate discriminator loss
	s_trg = style_encoder(x_ref_1, y_trg) // calculate style encode by first reference image and reference domain
	x_fake = generator(x_src, s_trg)	  // x_fake is generated image which contain reference image's style and content image
	out = discriminator(x_src, y_org)
	loss_adv = adv_loss(out, 1) // calculate discriminator loss

	// calculate style loss
	s_pred = style_encoder(x_fake, y_trg)  // style encode of generated image (x_fake) and reference domain
	loss_style = mean(abs(s_pred - s_trg)) // calculate style loss by style of first reference image (s_trg) style of generated image (x_fake) (s_pred)

	// calculate diversity sensitive loss
	s_trg2 = style_encoder(x_ref_2, y_trg) // calculate style encode by second ref image and ref domain
	x_fake2 = generator(x_src, s_trg2)	   // x_fake2 is generated image which contain reference 2 image's style and content image
	loss_ds = mean(abs(x_fake - x_fake2))  // calculate diversity sensitive loss from first style of first fake image and second fake image

	// calculate cycle-consistency loss
	s_org = style_encoder(x_src, x_trg) // calculate style encode by src image and src domain
	x_rec = generator(x_fake, s_org)	// x_rec is generated image which contain src image's style and content of x_src image
	loss_cyc = mean(abs(x_rec, x_src))	// calculate cycle-consistency loss from generated images (x_rec, x_src)

	loss = loss_adv + args.lambda_sty * loss_sty - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc // calculate total loss, lambda is passing from arguments
																				  return loss}

function discriminator(x, y)
{ // x: image; y domain
}

function style_encoder(x, y)
{ // x: image; y: domain
}

function generator(x, y)
{ // x: image; y: style encode
	x = from_rgb(x)
	// Image go through style encode
	for block in encode:
		x = block(x)
	for block in decode:
		x = block(x, s)
	return self.to_rgb(x) // Return generated image from image and style encode
}
