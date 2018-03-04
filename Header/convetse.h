#pragma once
#include <string.h>

int eff_histo[65536], histo_out[65536], Partial_sum[65536], g_histogram[65536];
void Plateau_Equalization16(unsigned short *Src, unsigned char *Des, int sx, int sy,
	unsigned long plateau)
{
	int i, img_len;;
	int Complete_sum;
	float scale, dum;


	img_len = sx*sy;
	//memset(g_histogram,0,65536*sizeof(unsigned long)); 
	memset(g_histogram, 0, 65536 * sizeof(unsigned int));
	//-------  Histogram calculation --------// 
	for (i = 0; i<img_len; i++)g_histogram[Src[i]]++;

	memset(Partial_sum, 0, 65536 * sizeof(int));
	for (i = 0; i<65536; i++) {
		if (g_histogram[i]<plateau)
			eff_histo[i] = g_histogram[i];
		else eff_histo[i] = plateau;
	}
	Partial_sum[0] = eff_histo[0];
	for (i = 1; i<65536; i++) {
		Partial_sum[i] = Partial_sum[i - 1] + eff_histo[i];
	}
	Complete_sum = Partial_sum[65535];
	scale = 255.f / Partial_sum[65535];

	for (i = 0; i<65536; i++) {
		dum = (scale* (float)(Partial_sum[i] - Partial_sum[0]));
		if (dum>255.f) dum = 255;
		if (dum<0.f) dum = 0;
		histo_out[i] = (unsigned char)dum;
	}

	for (i = 0; i<img_len; i++) Des[i] = histo_out[Src[i]];
}