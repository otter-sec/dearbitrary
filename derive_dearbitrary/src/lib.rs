use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::*;

fn expand_derive_dearbitrary(input: syn::DeriveInput) -> Result<TokenStream> {
    let dearbitrary_method = gen_dearbitrary_method(&input)?;

    let name = input.ident;
    let generics = add_trait_bounds(input.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    Ok(quote! {
        const _: () = {
            impl #impl_generics dearbitrary::Dearbitrary for #name #ty_generics #where_clause {
                #dearbitrary_method
            }
        };
    })
}

fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in generics.params.iter_mut() {
        if let GenericParam::Type(type_param) = param {
            type_param
                .bounds
                .push(parse_quote!(dearbitrary::Dearbitrary));
        }
    }
    generics
}

fn gen_dtor(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(names) => {
            let names = names.named.iter().enumerate().map(|(_, f)| {
                let name = f.ident.as_ref().unwrap();
                quote! { #name }
            });
            quote! { { #(#names,)* } }
        }
        Fields::Unnamed(names) => {
            let names = names.unnamed.iter().enumerate().map(|(i, _)| {
                let id = Ident::new(&format!("x{}", i), Span::call_site());
                quote! { #id }
            });
            quote! { ( #(#names),* ) }
        }
        Fields::Unit => quote!(),
    }
}

fn gen_dtor2_first(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(names) => {
            let n = names.named.len();
            let names = names.named.iter().enumerate().rev().map(|(i, f)| {
                let name = f.ident.as_ref().unwrap();
                if i == n - 1 {
                    quote! { _dearbitrator = #name.dearbitrary_first(); }
                } else {
                    quote! { #name.dearbitrary(&mut _dearbitrator); }
                }
            });
            quote! { { #(#names;)* } }
        }
        Fields::Unnamed(names) => {
            let n = names.unnamed.len();
            let names = names.unnamed.iter().enumerate().rev().map(|(i, _)| {
                let id = Ident::new(&format!("x{}", i), Span::call_site());
                if i == n - 1 {
                    quote! { _dearbitrator = #id.dearbitrary_first(); }
                } else {
                    quote! { #id.dearbitrary(&mut _dearbitrator); }
                }
            });
            quote! { { #(#names;)* } }
        }
        Fields::Unit => quote!( _dearbitrator = Dearbitrator::new(); ),
    }
}

fn gen_dtor2(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(names) => {
            let names = names.named.iter().enumerate().rev().map(|(_, f)| {
                let name = f.ident.as_ref().unwrap();
                quote! { #name.dearbitrary(_dearbitrator); }
            });
            quote! { { #(#names;)* } }
        }
        Fields::Unnamed(names) => {
            let names = names.unnamed.iter().enumerate().rev().map(|(i, _)| {
                let id = Ident::new(&format!("x{}", i), Span::call_site());
                quote! { #id.dearbitrary(_dearbitrator); }
            });
            quote! { { #(#names;)* } }
        }
        Fields::Unit => quote!(),
    }
}

fn gen_dearbitrary_method(input: &DeriveInput) -> Result<TokenStream> {
    let ident = &input.ident;
    let output = match &input.data {
        Data::Struct(data) => dearbitrary_structlike(&data.fields, ident),
        Data::Union(data) => dearbitrary_structlike(&Fields::Named(data.fields.clone()), ident),
        Data::Enum(data) => {
            let variants = data.variants.iter().enumerate().map(|(i, variant)| {
                let idx = i as u64;
                let dtor = gen_dtor(&variant.fields);
                let variant_name = &variant.ident;
                quote! { #ident::#variant_name #dtor => #idx }
            });
            let variants_first = variants.clone();

            let variants2 = data.variants.iter().enumerate().map(|(_, variant)| {
                let dtor = gen_dtor(&variant.fields);
                let dimpl = gen_dtor2(&variant.fields);

                let variant_name = &variant.ident;
                quote! { #ident::#variant_name #dtor => { #dimpl } }
            });

            let variants2_first = data.variants.iter().enumerate().map(|(_, variant)| {
                let dtor = gen_dtor(&variant.fields);
                let dimpl = gen_dtor2_first(&variant.fields);

                let variant_name = &variant.ident;
                quote! { #ident::#variant_name #dtor => { #dimpl } }
            });

            let count = data.variants.len() as u64;
            quote! {
                fn dearbitrary(&self, _dearbitrator: &mut dearbitrary::Dearbitrator) {
                    let val = match self {
                        #(#variants,)*
                        _ => unreachable!()
                    };
                    let mut x: u32 = ((val << 32) / #count ) as u32;
                    if ((u64::from(x) * #count) >> 32) < val {
                        x += 1;
                    }


                    match self {
                        #(#variants2,)*
                        _ => unreachable!()
                    };

                    x.dearbitrary(_dearbitrator);
                }

                fn dearbitrary_first(&self) -> dearbitrary::Dearbitrator {
                    let val = match self {
                        #(#variants_first,)*
                        _ => unreachable!()
                    };
                    let mut x: u32 = ((val << 32) / #count ) as u32;
                    if ((u64::from(x) * #count) >> 32) < val {
                        x += 1;
                    }

                    let mut _dearbitrator;

                    match self {
                        #(#variants2_first,)*
                        _ => unreachable!()
                    };

                    x.dearbitrary(&mut _dearbitrator);
                    _dearbitrator
                }
            }
        }
    };

    Ok(output)
}

fn dearbitrary_structlike(fields: &Fields, _ident: &Ident) -> TokenStream {
    let body = match fields {
        Fields::Named(names) => {
            let names: Vec<_> = names
                .named
                .iter()
                .rev()
                .enumerate()
                .map(|(_, f)| {
                    let name = f.ident.as_ref().unwrap();
                    quote! { self.#name.dearbitrary(_dearbitrator); }
                })
                .collect();
            quote! {
                #(
                    #names;
                )*
            }
        }
        Fields::Unnamed(names) => {
            let names: Vec<_> = names
                .unnamed
                .iter()
                .enumerate()
                .rev()
                .map(|(i, _)| {
                    let i = syn::Index::from(i);
                    quote! { self.#i.dearbitrary(_dearbitrator); }
                })
                .collect();
            quote! {
                #(
                    #names;
                )*
            }
        }
        Fields::Unit => quote! { ().dearbitrary(_dearbitrator); },
    };

    let body_first = match fields {
        Fields::Named(names) => {
            let names: Vec<_> = names
                .named
                .iter()
                .rev()
                .enumerate()
                .map(|(i, f)| {
                    let name = f.ident.as_ref().unwrap();
                    if i == 0 {
                        // first serialized
                        quote! { _dearbitrator = self.#name.dearbitrary_first(); }
                    } else {
                        quote! { self.#name.dearbitrary(&mut _dearbitrator); }
                    }
                })
                .collect();
            quote! {
                #(
                    #names;
                )*
            }
        }
        Fields::Unnamed(names) => {
            let n = names.unnamed.len();
            let names: Vec<_> = names
                .unnamed
                .iter()
                .enumerate()
                .rev()
                .map(|(i, _)| {
                    let syn_i = syn::Index::from(i);
                    if i == n - 1 {
                        // first serialized
                        quote! { _dearbitrator = self.#syn_i.dearbitrary_first(); }
                    } else {
                        quote! { self.#syn_i.dearbitrary(&mut _dearbitrator); }
                    }
                })
                .collect();
            quote! {
                #(
                    #names;
                )*
            }
        }
        Fields::Unit => quote! { _dearbitrator = ().dearbitrary_first(); },
    };

    quote! {
        fn dearbitrary(&self, _dearbitrator: &mut dearbitrary::Dearbitrator) {
            #body
        }

        fn dearbitrary_first(&self) -> dearbitrary::Dearbitrator {
            let mut _dearbitrator;
            #body_first
            _dearbitrator
        }
    }
}

#[proc_macro_derive(Dearbitrary, attributes(dearbitrary))]
pub fn derive_dearbitrary(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(tokens as syn::DeriveInput);
    expand_derive_dearbitrary(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
