from src.data_loaders.helper import *
from src.models.compgcn.compgcn_conv import CompGCNConv
from src.models.compgcn.compgcn_conv_basis import CompGCNConvBasis


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, num_rel, params=None, device = torch.device('cpu')):
		super(CompGCNBase, self).__init__(params)

		self.p.gcn_dim		= self.p.embed_dim if self.p.num_gcn_layers == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.node_feat_dim)).to(device)
		self.device		= device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.node_feat_dim)).to(self.device)
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.node_feat_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.node_feat_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.node_feat_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.num_gcn_layers == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.node_feat_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.num_gcn_layers == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2, node_index, edge_index, edge_type):

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		node_embed = self.init_embed[node_index]
		x, r	= self.conv1(node_embed.to(self.device), edge_index.to(self.device), edge_type.to(self.device), rel_embed=r.to(self.device))
		x	= drop1(x)
		x, r	= self.conv2(x, edge_index.to(self.device), edge_type.to(self.device), rel_embed=r) 	if self.p.num_gcn_layers == 2 else (x, r)
		x	= drop2(x) 							if self.p.num_gcn_layers == 2 else x

		x = x.to(self.device)
		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	#TODO : Update edge_index and edge_type with to(device)
	def __init__(self, params=None, device = torch.device('cpu')):
		super(self.__class__, self).__init__(params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.device = device

	def forward(self, x_id, edge_index, edge_type, sub, rel, batch_offsets):
		'''

		:param x_id: LongTensor of size [batch_size] of nodes in the batch of graphs for indexing into the node embedding table
		:param edge_index: Torch Geometric stacked edge indices with proper node ID offsets
		:param edge_type: Relation types corresponding to edge_index
		:param sub: LongTensor of shape [batch_size] of subject node IDs with proper offsets
		:param rel: LongTensor of shape [batch_size] of relation types
		:return:
		'''
		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop, x_id, edge_index, edge_type)
		obj_emb				= sub_emb + rel_emb # Get TransE representations for the neighbors
		# obj_emb: batch_size x embed_dim
		# obj_emb.unsqueeze(1): batch_size x 1 x embed_dim
		# all_ent: total_nodes_in_batch x embed_dim
		# x will have shape [batch_size x total_nodes_in_batch]
		# we only want scores corresponding to each batch
		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		x = x[batch_offsets, torch.arange(len(batch_offsets))]
		# For using BCEWithLogitsLoss
		# score	= torch.sigmoid(x)
		score = x

		return x, obj_emb

class CompGCN_DistMult(CompGCNBase):
	#TODO : Update edge_index and edge_type with to(device)
	def __init__(self, edge_index, edge_type, params=None, device = torch.device('cpu')):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.device = device

	def forward(self, sub, rel, edge_index, edge_type):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop, edge_index, edge_type)
		obj_emb = sub_emb * rel_emb # Get DistMult representations for the neighbors

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	# TODO : Update edge_index an edge_type with to(device)
	# TODO : Update (edge_index, edge_type) to the base function
	def __init__(self, edge_index, edge_type, params=None, device = torch.device('cpu')):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		self.device = device

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score